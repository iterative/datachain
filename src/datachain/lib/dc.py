import copy
import os
import os.path
import re
import sys
from collections.abc import Iterator, Sequence
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    ClassVar,
    Literal,
    Optional,
    TypeVar,
    Union,
    overload,
)

import orjson
import sqlalchemy
from pydantic import BaseModel
from sqlalchemy.sql.functions import GenericFunction
from tqdm import tqdm

from datachain.dataset import DatasetRecord
from datachain.func import literal
from datachain.func.base import Function
from datachain.func.func import Func
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.values_to_tuples import values_to_tuples
from datachain.lib.data_model import DataModel, DataType, DataValue, dict_to_data_model
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.file import (
    EXPORT_FILES_MAX_THREADS,
    ArrowRow,
    File,
    FileExporter,
    FileType,
    get_file_type,
)
from datachain.lib.file import ExportPlacement as FileExportPlacement
from datachain.lib.listing import get_file_info, get_listing, list_bucket, ls
from datachain.lib.listing_info import ListingInfo
from datachain.lib.meta_formats import read_meta
from datachain.lib.model_store import ModelStore
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import Aggregator, BatchMapper, Generator, Mapper, UDFBase
from datachain.lib.udf_signature import UdfSignature
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query import Session
from datachain.query.dataset import DatasetQuery, PartitionByType
from datachain.query.schema import DEFAULT_DELIMITER, Column, ColumnMeta
from datachain.sql.functions import path as pathfunc
from datachain.utils import batched_it, inside_notebook, row_to_nested_dict

if TYPE_CHECKING:
    import pandas as pd
    from pyarrow import DataType as ArrowDataType
    from typing_extensions import Concatenate, ParamSpec, Self

    from datachain.lib.hf import HFDatasetType

    P = ParamSpec("P")

C = Column

_T = TypeVar("_T")
D = TypeVar("D", bound="DataChain")
UDFObjT = TypeVar("UDFObjT", bound=UDFBase)

DEFAULT_PARQUET_CHUNK_SIZE = 100_000


def resolve_columns(
    method: "Callable[Concatenate[D, P], D]",
) -> "Callable[Concatenate[D, P], D]":
    """Decorator that resolvs input column names to their actual DB names. This is
    specially important for nested columns as user works with them by using dot
    notation e.g (file.name) but are actually defined with default delimiter
    in DB, e.g file__name.
    If there are any sql functions in arguments, they will just be transferred as is
    to a method.
    """

    @wraps(method)
    def _inner(self: D, *args: "P.args", **kwargs: "P.kwargs") -> D:
        resolved_args = self.signals_schema.resolve(
            *[arg for arg in args if not isinstance(arg, GenericFunction)]  # type: ignore[arg-type]
        ).db_signals()

        for idx, arg in enumerate(args):
            if isinstance(arg, GenericFunction):
                resolved_args.insert(idx, arg)  # type: ignore[arg-type]

        return method(self, *resolved_args, **kwargs)

    return _inner


class DatasetPrepareError(DataChainParamsError):  # noqa: D101
    def __init__(self, name, msg, output=None):  # noqa: D107
        name = f" '{name}'" if name else ""
        output = f" output '{output}'" if output else ""
        super().__init__(f"Dataset{name}{output} processing prepare error: {msg}")


class DatasetFromValuesError(DataChainParamsError):  # noqa: D101
    def __init__(self, name, msg):  # noqa: D107
        name = f" '{name}'" if name else ""
        super().__init__(f"Dataset{name} from values error: {msg}")


MergeColType = Union[str, Function, sqlalchemy.ColumnElement]


def _validate_merge_on(
    on: Union[MergeColType, Sequence[MergeColType]],
    ds: "DataChain",
) -> Sequence[MergeColType]:
    if isinstance(on, (str, sqlalchemy.ColumnElement)):
        return [on]
    if isinstance(on, Function):
        return [on.get_column(table=ds._query.table)]
    if isinstance(on, Sequence):
        return [
            c.get_column(table=ds._query.table) if isinstance(c, Function) else c
            for c in on
        ]


def _get_merge_error_str(col: MergeColType) -> str:
    if isinstance(col, str):
        return col
    if isinstance(col, Function):
        return f"{col.name}()"
    if isinstance(col, sqlalchemy.Column):
        return col.name.replace(DEFAULT_DELIMITER, ".")
    if isinstance(col, sqlalchemy.ColumnElement) and hasattr(col, "name"):
        return f"{col.name} expression"
    return str(col)


class DatasetMergeError(DataChainParamsError):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        on: Union[MergeColType, Sequence[MergeColType]],
        right_on: Optional[Union[MergeColType, Sequence[MergeColType]]],
        msg: str,
    ):
        def _get_str(
            on: Union[MergeColType, Sequence[MergeColType]],
        ) -> str:
            if not isinstance(on, Sequence):
                return str(on)  # type: ignore[unreachable]
            return ", ".join([_get_merge_error_str(col) for col in on])

        on_str = _get_str(on)
        right_on_str = (
            ", right_on='" + _get_str(right_on) + "'"
            if right_on and isinstance(right_on, Sequence)
            else ""
        )
        super().__init__(f"Merge error on='{on_str}'{right_on_str}: {msg}")


OutputType = Union[None, DataType, Sequence[str], dict[str, DataType]]


class Sys(DataModel):
    """Model for internal DataChain signals `id` and `rand`."""

    id: int
    rand: int


class DataChain:
    """DataChain - a data structure for batch data processing and evaluation.

    It represents a sequence of data manipulation steps such as reading data from
    storages, running AI or LLM models or calling external services API to validate or
    enrich data.

    Data in DataChain is presented as Python classes with arbitrary set of fields,
    including nested classes. The data classes have to inherit from `DataModel` class.
    The supported set of field types include: majority of the type supported by the
    underlyind library `Pydantic`.

    See Also:
        `DataChain.from_storage("s3://my-bucket/my-dir/")` - reading unstructured
            data files from storages such as S3, gs or Azure ADLS.

        `DataChain.save("name")` - saving to a dataset.

        `DataChain.from_dataset("name")` - reading from a dataset.

        `DataChain.from_values(fib=[1, 2, 3, 5, 8])` - generating from values.

        `DataChain.from_pandas(pd.DataFrame(...))` - generating from pandas.

        `DataChain.from_json("file.json")` - generating from json.

        `DataChain.from_csv("file.csv")` - generating from csv.

        `DataChain.from_parquet("file.parquet")` - generating from parquet.

    Example:
        ```py
        import os

        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage

        from datachain import DataChain, Column

        PROMPT = (
            "Was this bot dialog successful? "
            "Describe the 'result' as 'Yes' or 'No' in a short JSON"
        )

        model = "mistral-large-latest"
        api_key = os.environ["MISTRAL_API_KEY"]

        chain = (
            DataChain.from_storage("gs://datachain-demo/chatbot-KiT/")
            .limit(5)
            .settings(cache=True, parallel=5)
            .map(
                mistral_response=lambda file: MistralClient(api_key=api_key)
                .chat(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        ChatMessage(role="user", content=f"{PROMPT}: {file.read()}")
                    ],
                )
                .choices[0]
                .message.content,
            )
            .save()
        )

        try:
            print(chain.select("mistral_response").results())
        except Exception as e:
            print(f"do you have the right Mistral API key? {e}")
        ```
    """

    DEFAULT_FILE_RECORD: ClassVar[dict] = {
        "source": "",
        "path": "",
        "size": 0,
    }

    def __init__(
        self,
        query: DatasetQuery,
        settings: Settings,
        signal_schema: SignalSchema,
        setup: Optional[dict] = None,
        _sys: bool = False,
    ) -> None:
        """Don't instantiate this directly, use one of the from_XXX constructors."""
        self._query = query
        self._settings = settings
        self.signals_schema = signal_schema
        self._setup: dict = setup or {}
        self._sys = _sys

    @property
    def schema(self) -> dict[str, DataType]:
        """Get schema of the chain."""
        return self._effective_signals_schema.values

    def column(self, name: str) -> Column:
        """Returns Column instance with a type if name is found in current schema,
        otherwise raises an exception.
        """
        if "." in name:
            name_path = name.split(".")
        elif DEFAULT_DELIMITER in name:
            name_path = name.split(DEFAULT_DELIMITER)
        else:
            name_path = [name]
        for path, type_, _, _ in self.signals_schema.get_flat_tree():
            if path == name_path:
                return Column(name, python_to_sql(type_))

        raise ValueError(f"Column with name {name} not found in the schema")

    def c(self, column: Union[str, Column]) -> Column:
        """Returns Column instance attached to the current chain."""
        c = self.column(column) if isinstance(column, str) else self.column(column.name)
        c.table = self._query.table
        return c

    @property
    def session(self) -> Session:
        """Session of the chain."""
        return self._query.session

    @property
    def name(self) -> Optional[str]:
        """Name of the underlying dataset, if there is one."""
        return self._query.name

    @property
    def version(self) -> Optional[int]:
        """Version of the underlying dataset, if there is one."""
        return self._query.version

    @property
    def dataset(self) -> Optional[DatasetRecord]:
        """Underlying dataset, if there is one."""
        if not self.name:
            return None
        return self.session.catalog.get_dataset(self.name)

    def __or__(self, other: "Self") -> "Self":
        """Return `self.union(other)`."""
        return self.union(other)

    def print_schema(self) -> None:
        """Print schema of the chain."""
        self._effective_signals_schema.print_tree()

    def clone(self) -> "Self":
        """Make a copy of the chain in a new table."""
        return self._evolve(query=self._query.clone(new_table=True))

    def _evolve(
        self,
        *,
        query: Optional[DatasetQuery] = None,
        settings: Optional[Settings] = None,
        signal_schema=None,
        _sys=None,
    ) -> "Self":
        if query is None:
            query = self._query.clone(new_table=False)
        if settings is None:
            settings = self._settings
        if signal_schema is None:
            signal_schema = copy.deepcopy(self.signals_schema)
        if _sys is None:
            _sys = self._sys
        return type(self)(
            query, settings, signal_schema=signal_schema, setup=self._setup, _sys=_sys
        )

    def settings(
        self,
        cache=None,
        parallel=None,
        workers=None,
        min_task_size=None,
        prefetch: Optional[int] = None,
        sys: Optional[bool] = None,
    ) -> "Self":
        """Change settings for chain.

        This function changes specified settings without changing not specified ones.
        It returns chain, so, it can be chained later with next operation.

        Parameters:
            cache : data caching (default=False)
            parallel : number of thread for processors. True is a special value to
                enable all available CPUs (default=1)
            workers : number of distributed workers. Only for Studio mode. (default=1)
            min_task_size : minimum number of tasks (default=1)
            prefetch: number of workers to use for downloading files in advance.
                      This is enabled by default and uses 2 workers.
                      To disable prefetching, set it to 0.

        Example:
            ```py
            chain = (
                chain
                .settings(cache=True, parallel=8)
                .map(laion=process_webdataset(spec=WDSLaion), params="file")
            )
            ```
        """
        if sys is None:
            sys = self._sys
        settings = copy.copy(self._settings)
        settings.add(Settings(cache, parallel, workers, min_task_size, prefetch))
        return self._evolve(settings=settings, _sys=sys)

    def reset_settings(self, settings: Optional[Settings] = None) -> "Self":
        """Reset all settings to default values."""
        self._settings = settings if settings else Settings()
        return self

    def reset_schema(self, signals_schema: SignalSchema) -> "Self":  # noqa: D102
        self.signals_schema = signals_schema
        return self

    def add_schema(self, signals_schema: SignalSchema) -> "Self":  # noqa: D102
        self.signals_schema |= signals_schema
        return self

    @classmethod
    def from_storage(
        cls,
        uri: Union[str, os.PathLike[str]],
        *,
        type: FileType = "binary",
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        recursive: Optional[bool] = True,
        object_name: str = "file",
        update: bool = False,
        anon: bool = False,
        client_config: Optional[dict] = None,
    ) -> "Self":
        """Get data from a storage as a list of file with all file attributes.
        It returns the chain itself as usual.

        Parameters:
            uri : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            recursive : search recursively for the given path.
            object_name : Created object column name.
            update : force storage reindexing. Default is False.
            anon : If True, we will treat cloud bucket as public one
            client_config : Optional client configuration for the storage client.

        Example:
            Simple call from s3
            ```py
            chain = DataChain.from_storage("s3://my-bucket/my-dir")
            ```

            With AWS S3-compatible storage
            ```py
            chain = DataChain.from_storage(
                "s3://my-bucket/my-dir",
                client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
            )
            ```

            Pass existing session
            ```py
            session = Session.get()
            chain = DataChain.from_storage("s3://my-bucket/my-dir", session=session)
            ```
        """
        file_type = get_file_type(type)

        if anon:
            client_config = (client_config or {}) | {"anon": True}
        session = Session.get(session, client_config=client_config, in_memory=in_memory)
        cache = session.catalog.cache
        client_config = session.catalog.client_config

        list_ds_name, list_uri, list_path, list_ds_exists = get_listing(
            uri, session, update=update
        )

        # ds_name is None if object is a file, we don't want to use cache
        # or do listing in that case - just read that single object
        if not list_ds_name:
            dc = cls.from_values(
                session=session,
                settings=settings,
                in_memory=in_memory,
                file=[get_file_info(list_uri, cache, client_config=client_config)],
            )
            dc.signals_schema = dc.signals_schema.mutate({f"{object_name}": file_type})
            return dc

        if update or not list_ds_exists:
            # disable prefetch for listing, as it pre-downloads all files
            (
                cls.from_records(
                    DataChain.DEFAULT_FILE_RECORD,
                    session=session,
                    settings=settings,
                    in_memory=in_memory,
                )
                .settings(prefetch=0)
                .gen(
                    list_bucket(list_uri, cache, client_config=client_config),
                    output={f"{object_name}": File},
                )
                .save(list_ds_name, listing=True)
            )

        dc = cls.from_dataset(list_ds_name, session=session, settings=settings)
        dc.signals_schema = dc.signals_schema.mutate({f"{object_name}": file_type})

        return ls(dc, list_path, recursive=recursive, object_name=object_name)

    @classmethod
    def from_dataset(
        cls,
        name: str,
        version: Optional[int] = None,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        fallback_to_studio: bool = True,
    ) -> "Self":
        """Get data from a saved Dataset. It returns the chain itself.
        If dataset or version is not found locally, it will try to pull it from Studio.

        Parameters:
            name : dataset name
            version : dataset version
            session : Session to use for the chain.
            settings : Settings to use for the chain.
            fallback_to_studio : Try to pull dataset from Studio if not found locally.
                Default is True.

        Example:
            ```py
            chain = DataChain.from_dataset("my_cats")
            ```

            ```py
            chain = DataChain.from_dataset("my_cats", fallback_to_studio=False)
            ```

            ```py
            chain = DataChain.from_dataset("my_cats", version=1)
            ```

            ```py
            session = Session.get(client_config={"aws_endpoint_url": "<minio-url>"})
            settings = {
                "cache": True,
                "parallel": 4,
                "workers": 4,
                "min_task_size": 1000,
                "prefetch": 10,
            }
            chain = DataChain.from_dataset(
                name="my_cats",
                version=1,
                session=session,
                settings=settings,
                fallback_to_studio=True,
            )
            ```
        """
        from datachain.telemetry import telemetry

        query = DatasetQuery(
            name=name,
            version=version,
            session=session,
            indexing_column_types=File._datachain_column_types,
            fallback_to_studio=fallback_to_studio,
        )
        telemetry.send_event_once("class", "datachain_init", name=name, version=version)
        if settings:
            _settings = Settings(**settings)
        else:
            _settings = Settings()

        signals_schema = SignalSchema({"sys": Sys})
        if query.feature_schema:
            signals_schema |= SignalSchema.deserialize(query.feature_schema)
        else:
            signals_schema |= SignalSchema.from_column_types(query.column_types or {})
        return cls(query, _settings, signals_schema)

    @classmethod
    def from_json(
        cls,
        path: Union[str, os.PathLike[str]],
        type: FileType = "text",
        spec: Optional[DataType] = None,
        schema_from: Optional[str] = "auto",
        jmespath: Optional[str] = None,
        object_name: Optional[str] = "",
        model_name: Optional[str] = None,
        format: Optional[str] = "json",
        nrows=None,
        **kwargs,
    ) -> "DataChain":
        """Get data from JSON. It returns the chain itself.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "text".
            spec : optional Data Model
            schema_from : path to sample to infer spec (if schema not provided)
            object_name : generated object column name
            model_name : optional generated model name
            format: "json", "jsonl"
            jmespath : optional JMESPATH expression to reduce JSON
            nrows : optional row limit for jsonl and JSON arrays

        Example:
            infer JSON schema from data, reduce using JMESPATH
            ```py
            chain = DataChain.from_json("gs://json", jmespath="key1.key2")
            ```

            infer JSON schema from a particular path
            ```py
            chain = DataChain.from_json("gs://json_ds", schema_from="gs://json/my.json")
            ```
        """
        if schema_from == "auto":
            schema_from = str(path)

        def jmespath_to_name(s: str):
            name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
            return s[:name_end]

        if (not object_name) and jmespath:
            object_name = jmespath_to_name(jmespath)
        if not object_name:
            object_name = format
        chain = DataChain.from_storage(uri=path, type=type, **kwargs)
        signal_dict = {
            object_name: read_meta(
                schema_from=schema_from,
                format=format,
                spec=spec,
                model_name=model_name,
                jmespath=jmespath,
                nrows=nrows,
            )
        }
        # disable prefetch if nrows is set
        settings = {"prefetch": 0} if nrows else {}
        return chain.settings(**settings).gen(**signal_dict)  # type: ignore[misc, arg-type]

    def explode(
        self,
        col: str,
        model_name: Optional[str] = None,
        object_name: Optional[str] = None,
        schema_sample_size: int = 1,
    ) -> "DataChain":
        """Explodes a column containing JSON objects (dict or str DataChain type) into
           individual columns based on the schema of the JSON. Schema is inferred from
           the first row of the column.

        Args:
            col: the name of the column containing JSON to be exploded.
            model_name: optional generated model name.  By default generates the name
                automatically.
            object_name: optional generated object column name. By default generates the
                name automatically.
            schema_sample_size: the number of rows to use for inferring the schema of
                the JSON (in case some fields are optional and it's not enough to
                analyze a single row).

        Returns:
            DataChain: A new DataChain instance with the new set of columns.
        """
        import json

        import pyarrow as pa

        from datachain.lib.arrow import schema_to_output

        json_values = list(self.limit(schema_sample_size).collect(col))
        json_dicts = [
            json.loads(json_value) if isinstance(json_value, str) else json_value
            for json_value in json_values
        ]

        if any(not isinstance(json_dict, dict) for json_dict in json_dicts):
            raise TypeError(f"Column {col} should be a string or dict type with JSON")

        schema = pa.Table.from_pylist(json_dicts).schema
        output, original_names = schema_to_output(schema, None)

        if not model_name:
            model_name = f"{col.title()}ExplodedModel"

        model = dict_to_data_model(model_name, output, original_names)

        def json_to_model(json_value: Union[str, dict]):
            json_dict = (
                json.loads(json_value) if isinstance(json_value, str) else json_value
            )
            return model.model_validate(json_dict)

        if not object_name:
            object_name = f"{col}_expl"

        return self.map(json_to_model, params=col, output={object_name: model})

    @classmethod
    def datasets(
        cls,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        object_name: str = "dataset",
        include_listing: bool = False,
        studio: bool = False,
    ) -> "DataChain":
        """Generate chain with list of registered datasets.

        Args:
            session: Optional session instance. If not provided, uses default session.
            settings: Optional dictionary of settings to configure the chain.
            in_memory: If True, creates an in-memory session. Defaults to False.
            object_name: Name of the output object in the chain. Defaults to "dataset".
            include_listing: If True, includes listing datasets. Defaults to False.
            studio: If True, returns datasets from Studio only,
                otherwise returns all local datasets. Defaults to False.

        Returns:
            DataChain: A new DataChain instance containing dataset information.

        Example:
            ```py
            from datachain import DataChain

            chain = DataChain.datasets()
            for ds in chain.collect("dataset"):
                print(f"{ds.name}@v{ds.version}")
            ```
        """
        session = Session.get(session, in_memory=in_memory)
        catalog = session.catalog

        datasets = [
            DatasetInfo.from_models(d, v, j)
            for d, v, j in catalog.list_datasets_versions(
                include_listing=include_listing, studio=studio
            )
        ]

        return cls.from_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            output={object_name: DatasetInfo},
            **{object_name: datasets},  # type: ignore[arg-type]
        )

    @classmethod
    def listings(
        cls,
        session: Optional[Session] = None,
        in_memory: bool = False,
        object_name: str = "listing",
        **kwargs,
    ) -> "DataChain":
        """Generate chain with list of cached listings.
        Listing is a special kind of dataset which has directory listing data of
        some underlying storage (e.g S3 bucket).

        Example:
            ```py
            from datachain import DataChain
            DataChain.listings().show()
            ```
        """
        session = Session.get(session, in_memory=in_memory)
        catalog = kwargs.get("catalog") or session.catalog

        return cls.from_values(
            session=session,
            in_memory=in_memory,
            output={object_name: ListingInfo},
            **{object_name: catalog.listings()},  # type: ignore[arg-type]
        )

    def save(  # type: ignore[override]
        self, name: Optional[str] = None, version: Optional[int] = None, **kwargs
    ) -> "Self":
        """Save to a Dataset. It returns the chain itself.

        Parameters:
            name : dataset name. Empty name saves to a temporary dataset that will be
                removed after process ends. Temp dataset are useful for optimization.
            version : version of a dataset. Default - the last version that exist.
        """
        schema = self.signals_schema.clone_without_sys_signals().serialize()
        return self._evolve(
            query=self._query.save(
                name=name, version=version, feature_schema=schema, **kwargs
            )
        )

    def apply(self, func, *args, **kwargs):
        """Apply any function to the chain.

        Useful for reusing in a chain of operations.

        Example:
            ```py
            def parse_stem(chain):
                return chain.map(
                    lambda file: file.get_file_stem()
                    output={"stem": str}
                )

            chain = (
                DataChain.from_storage("s3://my-bucket")
                .apply(parse_stem)
                .filter(C("stem").glob("*cat*"))
            )
            ```
        """
        return func(self, *args, **kwargs)

    def map(
        self,
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: OutputType = None,
        **signal_map,
    ) -> "Self":
        """Apply a function to each row to create new signals. The function should
        return a new object for each row. It returns a chain itself with new signals.

        Input-output relationship: 1:1

        Parameters:
            func : Function applied to each row.
            params : List of column names used as input for the function. Default
                    is taken from function signature.
            output : Dictionary defining new signals and their corresponding types.
                    Default type is taken from function signature. Default can be also
                    taken from kwargs - **signal_map (see below).
                    If signal name is defined using signal_map (see below) only a single
                    type value can be used.
            **signal_map : kwargs can be used to define `func` together with it's return
                    signal name in format of `map(my_sign=my_func)`. This helps define
                    signal names and function in a nicer way.

        Example:
            Using signal_map and single type in output:
            ```py
            chain = chain.map(value=lambda name: name[:-4] + ".json", output=str)
            chain.save("new_dataset")
            ```

            Using func and output as a map:
            ```py
            chain = chain.map(
                lambda name: name.split("."), output={"stem": str, "ext": str}
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Mapper, func, params, output, signal_map)
        if (prefetch := self._settings.prefetch) is not None:
            udf_obj.prefetch = prefetch

        return self._evolve(
            query=self._query.add_signals(
                udf_obj.to_udf_wrapper(),
                **self._settings.to_dict(),
            ),
            signal_schema=self.signals_schema | udf_obj.output,
        )

    def gen(
        self,
        func: Optional[Union[Callable, Generator]] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: OutputType = None,
        **signal_map,
    ) -> "Self":
        r"""Apply a function to each row to create new rows (with potentially new
        signals). The function needs to return a new objects for each of the new rows.
        It returns a chain itself with new signals.

        Input-output relationship: 1:N

        This method is similar to `map()`, uses the same list of parameters, but with
        one key differences: It produces a sequence of rows for each input row (like
        extracting multiple file records from a single tar file or bounding boxes from a
        single image file).

        Example:
            ```py
            chain = chain.gen(
                line=lambda file: [l for l in file.read().split("\n")],
                output=str,
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Generator, func, params, output, signal_map)
        if (prefetch := self._settings.prefetch) is not None:
            udf_obj.prefetch = prefetch
        return self._evolve(
            query=self._query.generate(
                udf_obj.to_udf_wrapper(),
                **self._settings.to_dict(),
            ),
            signal_schema=udf_obj.output,
        )

    def agg(
        self,
        func: Optional[Callable] = None,
        partition_by: Optional[PartitionByType] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: OutputType = None,
        **signal_map,
    ) -> "Self":
        """Aggregate rows using `partition_by` statement and apply a function to the
        groups of aggregated rows. The function needs to return new objects for each
        group of the new rows. It returns a chain itself with new signals.

        Input-output relationship: N:M

        This method bears similarity to `gen()` and `map()`, employing a comparable set
        of parameters, yet differs in two crucial aspects:
        1. The `partition_by` parameter: This specifies the column name or a list of
           column names that determine the grouping criteria for aggregation.
        2. Group-based UDF function input: Instead of individual rows, the function
           receives a list all rows within each group defined by `partition_by`.

        Examples:
            ```py
            chain = chain.agg(
                total=lambda category, amount: [sum(amount)],
                output=float,
                partition_by="category",
            )
            chain.save("new_dataset")
            ```

            An alternative syntax, when you need to specify a more complex function:

            ```py
            # It automatically resolves which columns to pass to the function
            # by looking at the function signature.
            def agg_sum(
                file: list[File], amount: list[float]
            ) -> Iterator[tuple[File, float]]:
                yield file[0], sum(amount)

            chain = chain.agg(
                agg_sum,
                output={"file": File, "total": float},
                # Alternative syntax is to use `C` (short for Column) to specify
                # a column name or a nested column, e.g. C("file.path").
                partition_by=C("category"),
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Aggregator, func, params, output, signal_map)
        return self._evolve(
            query=self._query.generate(
                udf_obj.to_udf_wrapper(),
                partition_by=partition_by,
                **self._settings.to_dict(),
            ),
            signal_schema=udf_obj.output,
        )

    def batch_map(
        self,
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: OutputType = None,
        batch: int = 1000,
        **signal_map,
    ) -> "Self":
        """This is a batch version of `map()`.

        Input-output relationship: N:N

        It accepts the same parameters plus an
        additional parameter:

            batch : Size of each batch passed to `func`. Defaults to 1000.

        Example:
            ```py
            chain = chain.batch_map(
                sqrt=lambda size: np.sqrt(size),
                output=float
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(BatchMapper, func, params, output, signal_map)
        return self._evolve(
            query=self._query.add_signals(
                udf_obj.to_udf_wrapper(batch),
                **self._settings.to_dict(),
            ),
            signal_schema=self.signals_schema | udf_obj.output,
        )

    def _udf_to_obj(
        self,
        target_class: type[UDFObjT],
        func: Optional[Union[Callable, UDFObjT]],
        params: Union[None, str, Sequence[str]],
        output: OutputType,
        signal_map,
    ) -> UDFObjT:
        is_generator = target_class.is_output_batched
        name = self.name or ""

        sign = UdfSignature.parse(name, signal_map, func, params, output, is_generator)
        DataModel.register(list(sign.output_schema.values.values()))

        signals_schema = self.signals_schema
        if self._sys:
            signals_schema = SignalSchema({"sys": Sys}) | signals_schema

        params_schema = signals_schema.slice(sign.params, self._setup)

        return target_class._create(sign, params_schema)

    def _extend_to_data_model(self, method_name, *args, **kwargs):
        query_func = getattr(self._query, method_name)

        new_schema = self.signals_schema.resolve(*args)
        columns = [C(col) for col in new_schema.db_signals()]
        return query_func(*columns, **kwargs)

    @resolve_columns
    def order_by(self, *args, descending: bool = False) -> "Self":
        """Orders by specified set of columns.

        Parameters:
            descending (bool): Whether to sort in descending order or not.

        Example:
            ```py
            dc.order_by("similarity_score", descending=True).limit(10)
            ```

        Note:
            Order is not guaranteed when steps are added after an `order_by` statement.
            I.e. when using `from_dataset` an `order_by` statement should be used if
            the order of the records in the chain is important.
            Using `order_by` directly before `limit`, `collect` and `collect_flatten`
            will give expected results.
            See https://github.com/iterative/datachain/issues/477 for further details.
        """
        if descending:
            args = tuple(sqlalchemy.desc(a) for a in args)

        return self._evolve(query=self._query.order_by(*args))

    def distinct(self, arg: str, *args: str) -> "Self":  # type: ignore[override]
        """Removes duplicate rows based on uniqueness of some input column(s)
        i.e if rows are found with the same value of input column(s), only one
        row is left in the result set.

        Example:
            ```py
            dc.distinct("file.parent", "file.name")
            ```
        """
        return self._evolve(
            query=self._query.distinct(
                *self.signals_schema.resolve(arg, *args).db_signals()
            )
        )

    def select(self, *args: str, _sys: bool = True) -> "Self":
        """Select only a specified set of signals."""
        new_schema = self.signals_schema.resolve(*args)
        if self._sys and _sys:
            new_schema = SignalSchema({"sys": Sys}) | new_schema
        columns = new_schema.db_signals()
        return self._evolve(
            query=self._query.select(*columns), signal_schema=new_schema
        )

    def select_except(self, *args: str) -> "Self":
        """Select all the signals expect the specified signals."""
        new_schema = self.signals_schema.select_except_signals(*args)
        columns = new_schema.db_signals()
        return self._evolve(
            query=self._query.select(*columns), signal_schema=new_schema
        )

    def group_by(
        self,
        *,
        partition_by: Optional[Union[str, Func, Sequence[Union[str, Func]]]] = None,
        **kwargs: Func,
    ) -> "Self":
        """Group rows by specified set of signals and return new signals
        with aggregated values.

        The supported functions:
           count(), sum(), avg(), min(), max(), any_value(), collect(), concat()

        Example:
            ```py
            chain = chain.group_by(
                cnt=func.count(),
                partition_by=("file_source", "file_ext"),
            )
            ```
        """
        if partition_by is None:
            partition_by = []
        elif isinstance(partition_by, (str, Func)):
            partition_by = [partition_by]

        partition_by_columns: list[Column] = []
        signal_columns: list[Column] = []
        schema_fields: dict[str, DataType] = {}
        keep_columns: list[str] = []

        # validate partition_by columns and add them to the schema
        for col in partition_by:
            if isinstance(col, str):
                col_db_name = ColumnMeta.to_db_name(col)
                col_type = self.signals_schema.get_column_type(col_db_name)
                column = Column(col_db_name, python_to_sql(col_type))
                if col not in keep_columns:
                    keep_columns.append(col)
            elif isinstance(col, Function):
                column = col.get_column(self.signals_schema)
                col_db_name = column.name
                col_type = column.type.python_type
                schema_fields[col_db_name] = col_type
            else:
                raise DataChainColumnError(
                    col,
                    (
                        f"partition_by column {col} has type {type(col)}"
                        " but expected str or Function"
                    ),
                )
            partition_by_columns.append(column)

        # validate signal columns and add them to the schema
        if not kwargs:
            raise ValueError("At least one column should be provided for group_by")
        for col_name, func in kwargs.items():
            if not isinstance(func, Func):
                raise DataChainColumnError(
                    col_name,
                    f"Column {col_name} has type {type(func)} but expected Func object",
                )
            column = func.get_column(self.signals_schema, label=col_name)
            signal_columns.append(column)
            schema_fields[col_name] = func.get_result_type(self.signals_schema)

        signal_schema = SignalSchema(schema_fields)
        if keep_columns:
            signal_schema |= self.signals_schema.to_partial(*keep_columns)

        return self._evolve(
            query=self._query.group_by(signal_columns, partition_by_columns),
            signal_schema=signal_schema,
        )

    def mutate(self, **kwargs) -> "Self":
        """Create new signals based on existing signals.

        This method cannot modify existing columns. If you need to modify an
        existing column, use a different name for the new column and then use
        `select()` to choose which columns to keep.

        This method is vectorized and more efficient compared to map(), and it does not
        extract or download any data from the internal database. However, it can only
        utilize predefined built-in functions and their combinations.

        The supported functions:
           Numerical:   +, -, *, /, rand(), avg(), count(), func(),
                        greatest(), least(), max(), min(), sum()
           String:      length(), split(), replace(), regexp_replace()
           Filename:    name(), parent(), file_stem(), file_ext()
           Array:       length(), sip_hash_64(), euclidean_distance(),
                        cosine_distance()
           Window:      row_number(), rank(), dense_rank(), first()

        Example:
        ```py
         dc.mutate(
            area=Column("image.height") * Column("image.width"),
            extension=file_ext(Column("file.name")),
            dist=cosine_distance(embedding_text, embedding_image)
        )
        ```

        Window function example:
        ```py
        window = func.window(partition_by="file.parent", order_by="file.size")
        dc.mutate(
            row_number=func.row_number().over(window),
        )
        ```

        This method can be also used to rename signals. If the Column("name") provided
        as value for the new signal - the old column will be dropped. Otherwise a new
        column is created.

        Example:
        ```py
         dc.mutate(
            newkey=Column("oldkey")
        )
        ```
        """
        from sqlalchemy.sql.sqltypes import NullType

        primitives = (bool, str, int, float)

        for col_name, expr in kwargs.items():
            if not isinstance(expr, (*primitives, Column, Func)) and isinstance(
                expr.type, NullType
            ):
                raise DataChainColumnError(
                    col_name, f"Cannot infer type with expression {expr}"
                )

        mutated = {}
        schema = self.signals_schema
        for name, value in kwargs.items():
            if isinstance(value, Column):
                # renaming existing column
                for signal in schema.db_signals(name=value.name, as_columns=True):
                    mutated[signal.name.replace(value.name, name, 1)] = signal  # type: ignore[union-attr]
            elif isinstance(value, Func):
                # adding new signal
                mutated[name] = value.get_column(schema)
            elif isinstance(value, primitives):
                # adding simple python constant primitives like str, int, float, bool
                val = literal(value)
                val.type = python_to_sql(type(value))()
                mutated[name] = val  # type: ignore[assignment]
            else:
                # adding new signal
                mutated[name] = value

        return self._evolve(
            query=self._query.mutate(**mutated), signal_schema=schema.mutate(kwargs)
        )

    @property
    def _effective_signals_schema(self) -> "SignalSchema":
        """Effective schema used for user-facing API like collect, to_pandas, etc."""
        signals_schema = self.signals_schema
        if not self._sys:
            return signals_schema.clone_without_sys_signals()
        return signals_schema

    @overload
    def collect_flatten(self) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def collect_flatten(self, *, include_hidden: bool) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def collect_flatten(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> Iterator[_T]: ...

    @overload
    def collect_flatten(
        self,
        *,
        row_factory: Callable[[list[str], tuple[Any, ...]], _T],
        include_hidden: bool,
    ) -> Iterator[_T]: ...

    def collect_flatten(self, *, row_factory=None, include_hidden: bool = True):
        """Yields flattened rows of values as a tuple.

        Args:
            row_factory : A callable to convert row to a custom format.
                          It should accept two arguments: a list of column names and
                          a tuple of row values.
            include_hidden: Whether to include hidden signals from the schema.
        """
        db_signals = self._effective_signals_schema.db_signals(
            include_hidden=include_hidden
        )
        with self._query.ordered_select(*db_signals).as_iterable() as rows:
            if row_factory:
                rows = (row_factory(db_signals, r) for r in rows)  # type: ignore[assignment]
            yield from rows

    def to_columnar_data_with_names(
        self, chunk_size: int = DEFAULT_PARQUET_CHUNK_SIZE
    ) -> tuple[list[str], Iterator[list[list[Any]]]]:
        """Returns column names and the results as an iterator that provides chunks,
        with each chunk containing a list of columns, where each column contains a
        list of the row values for that column in that chunk. Useful for columnar data
        formats, such as parquet or other OLAP databases.
        """
        headers, _ = self._effective_signals_schema.get_headers_with_length()
        column_names = [".".join(filter(None, header)) for header in headers]

        results_iter = self.collect_flatten()

        def column_chunks() -> Iterator[list[list[Any]]]:
            for chunk_iter in batched_it(results_iter, chunk_size):
                columns: list[list[Any]] = [[] for _ in column_names]
                for row in chunk_iter:
                    for i, col in enumerate(columns):
                        col.append(row[i])
                yield columns

        return column_names, column_chunks()

    @overload
    def results(self) -> list[tuple[Any, ...]]: ...

    @overload
    def results(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> list[_T]: ...

    @overload
    def results(
        self,
        *,
        row_factory: Callable[[list[str], tuple[Any, ...]], _T],
        include_hidden: bool,
    ) -> list[_T]: ...

    @overload
    def results(self, *, include_hidden: bool) -> list[tuple[Any, ...]]: ...

    def results(self, *, row_factory=None, include_hidden=True):  # noqa: D102
        if row_factory is None:
            return list(self.collect_flatten(include_hidden=include_hidden))
        return list(
            self.collect_flatten(row_factory=row_factory, include_hidden=include_hidden)
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Convert every row to a dictionary."""

        def to_dict(cols: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
            return dict(zip(cols, row))

        return self.results(row_factory=to_dict)

    @overload
    def collect(self) -> Iterator[tuple[DataValue, ...]]: ...

    @overload
    def collect(self, col: str) -> Iterator[DataValue]: ...

    @overload
    def collect(self, *cols: str) -> Iterator[tuple[DataValue, ...]]: ...

    def collect(self, *cols: str) -> Iterator[Union[DataValue, tuple[DataValue, ...]]]:  # type: ignore[overload-overlap,misc]
        """Yields rows of values, optionally limited to the specified columns.

        Args:
            *cols: Limit to the specified columns. By default, all columns are selected.

        Yields:
            (DataType): Yields a single item if a column is selected.
            (tuple[DataType, ...]): Yields a tuple of items if multiple columns are
                selected.

        Example:
            Iterating over all rows:
            ```py
            for row in dc.collect():
                print(row)
            ```

            Iterating over all rows with selected columns:
            ```py
            for name, size in dc.collect("file.name", "file.size"):
                print(name, size)
            ```

            Iterating over a single column:
            ```py
            for file in dc.collect("file.name"):
                print(file)
            ```
        """
        chain = self.select(*cols) if cols else self
        signals_schema = chain._effective_signals_schema
        db_signals = signals_schema.db_signals()
        with self._query.ordered_select(*db_signals).as_iterable() as rows:
            for row in rows:
                ret = signals_schema.row_to_features(
                    row, catalog=chain.session.catalog, cache=chain._settings.cache
                )
                yield ret[0] if len(cols) == 1 else tuple(ret)

    def to_pytorch(
        self,
        transform=None,
        tokenizer=None,
        tokenizer_kwargs=None,
        num_samples=0,
        remove_prefetched: bool = False,
    ):
        """Convert to pytorch dataset format.

        Args:
            transform (Transform): Torchvision transforms to apply to the dataset.
            tokenizer (Callable): Tokenizer to use to tokenize text values.
            tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
            num_samples (int): Number of random samples to draw for each epoch.
                This argument is ignored if `num_samples=0` (the default).
            remove_prefetched (bool): Whether to remove prefetched files after reading.

        Example:
            ```py
            from torch.utils.data import DataLoader
            loader = DataLoader(
                chain.select("file", "label").to_pytorch(),
                batch_size=16
            )
            ```
        """
        from datachain.torch import PytorchDataset

        if self._query.attached:
            chain = self
        else:
            chain = self.save()
        assert chain.name is not None  # for mypy
        return PytorchDataset(
            chain.name,
            chain.version,
            catalog=self.session.catalog,
            transform=transform,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            num_samples=num_samples,
            dc_settings=chain._settings,
            remove_prefetched=remove_prefetched,
        )

    def remove_file_signals(self) -> "Self":  # noqa: D102
        schema = self.signals_schema.clone_without_file_signals()
        return self.select(*schema.values.keys())

    def merge(
        self,
        right_ds: "DataChain",
        on: Union[MergeColType, Sequence[MergeColType]],
        right_on: Optional[Union[MergeColType, Sequence[MergeColType]]] = None,
        inner=False,
        full=False,
        rname="right_",
    ) -> "Self":
        """Merge two chains based on the specified criteria.

        Parameters:
            right_ds: Chain to join with.
            on: Predicate ("column.name", C("column.name"), or Func) or list of
                Predicates to join on. If both chains have the same predicates then
                this predicate is enough for the join. Otherwise, `right_on` parameter
                has to specify the predicates for the other chain.
            right_on: Optional predicate or list of Predicates for the `right_ds`
                to join.
            inner (bool): Whether to run inner join or outer join.
            full (bool): Whether to run full outer join.
            rname (str): Name prefix for conflicting signal names.

        Examples:
            ```py
            meta = meta_emd.merge(meta_pq, on=(C.name, C.emd__index),
                                  right_on=(C.name, C.pq__index))
            ```

            ```py
            imgs.merge(captions,
                       on=func.path.file_stem(imgs.c("file.path")),
                       right_on=func.path.file_stem(captions.c("file.path"))
            ```
        )
        """
        if on is None:
            raise DatasetMergeError(["None"], None, "'on' must be specified")

        on = _validate_merge_on(on, self)
        if not on:
            raise DatasetMergeError(
                on,
                right_on,
                (
                    "'on' must be 'str', 'Func' or 'Sequence' object "
                    f"but got type '{type(on)}'"
                ),
            )

        if right_on is not None:
            right_on = _validate_merge_on(right_on, right_ds)
            if not right_on:
                raise DatasetMergeError(
                    on,
                    right_on,
                    "'right_on' must be 'str', 'Func' or 'Sequence' object"
                    f" but got type '{type(right_on)}'",
                )

            if len(right_on) != len(on):
                raise DatasetMergeError(
                    on, right_on, "'on' and 'right_on' must have the same length'"
                )

        if self == right_ds:
            right_ds = right_ds.clone()

        errors = []

        def _resolve(
            ds: DataChain,
            col: Union[str, Function, sqlalchemy.ColumnElement],
            side: Union[str, None],
        ):
            try:
                if isinstance(col, Function):
                    return ds.c(col.get_column())
                return ds.c(col) if isinstance(col, (str, C)) else col
            except ValueError:
                if side:
                    errors.append(f"{_get_merge_error_str(col)} in {side}")

        ops = [
            _resolve(self, left, "left")
            == _resolve(right_ds, right, "right" if right_on else None)
            for left, right in zip(on, right_on or on)
        ]

        if errors:
            raise DatasetMergeError(
                on, right_on, f"Could not resolve {', '.join(errors)}"
            )

        query = self._query.join(
            right_ds._query, sqlalchemy.and_(*ops), inner, full, rname + "{name}"
        )
        query.feature_schema = None
        ds = self._evolve(query=query)

        signals_schema = self.signals_schema.clone_without_sys_signals()
        right_signals_schema = right_ds.signals_schema.clone_without_sys_signals()
        ds.signals_schema = SignalSchema({"sys": Sys}) | signals_schema.merge(
            right_signals_schema, rname
        )

        return ds

    def union(self, other: "Self") -> "Self":
        """Return the set union of the two datasets.

        Parameters:
            other: chain whose rows will be added to `self`.
        """
        return self._evolve(query=self._query.union(other._query))

    def subtract(  # type: ignore[override]
        self,
        other: "DataChain",
        on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
    ) -> "Self":
        """Remove rows that appear in another chain.

        Parameters:
            other: chain whose rows will be removed from `self`
            on: columns to consider for determining row equality in `self`.
                If unspecified, defaults to all common columns
                between `self` and `other`.
            right_on: columns to consider for determining row equality in `other`.
                If unspecified, defaults to the same values as `on`.
        """
        if isinstance(on, str):
            if not on:
                raise DataChainParamsError("'on' cannot be an empty string")
            on = [on]
        elif isinstance(on, Sequence):
            if not on or any(not col for col in on):
                raise DataChainParamsError("'on' cannot contain empty strings")

        if isinstance(right_on, str):
            if not right_on:
                raise DataChainParamsError("'right_on' cannot be an empty string")
            right_on = [right_on]
        elif isinstance(right_on, Sequence):
            if not right_on or any(not col for col in right_on):
                raise DataChainParamsError("'right_on' cannot contain empty strings")

        if on is None and right_on is None:
            other_columns = set(other._effective_signals_schema.db_signals())
            signals = [
                c
                for c in self._effective_signals_schema.db_signals()
                if c in other_columns
            ]
            if not signals:
                raise DataChainParamsError("subtract(): no common columns")
        elif on is not None and right_on is None:
            right_on = on
            signals = list(self.signals_schema.resolve(*on).db_signals())
        elif on is None and right_on is not None:
            raise DataChainParamsError(
                "'on' must be specified when 'right_on' is provided"
            )
        else:
            if not isinstance(on, Sequence) or not isinstance(right_on, Sequence):
                raise TypeError(
                    "'on' and 'right_on' must be 'str' or 'Sequence' object"
                )
            if len(on) != len(right_on):
                raise DataChainParamsError(
                    "'on' and 'right_on' must have the same length"
                )
            signals = list(
                zip(
                    self.signals_schema.resolve(*on).db_signals(),
                    other.signals_schema.resolve(*right_on).db_signals(),
                )  # type: ignore[arg-type]
            )
        return self._evolve(query=self._query.subtract(other._query, signals))  # type: ignore[arg-type]

    def compare(
        self,
        other: "DataChain",
        on: Union[str, Sequence[str]],
        right_on: Optional[Union[str, Sequence[str]]] = None,
        compare: Optional[Union[str, Sequence[str]]] = None,
        right_compare: Optional[Union[str, Sequence[str]]] = None,
        added: bool = True,
        deleted: bool = True,
        modified: bool = True,
        same: bool = False,
        status_col: Optional[str] = None,
    ) -> "DataChain":
        """Comparing two chains by identifying rows that are added, deleted, modified
        or same. Result is the new chain that has additional column with possible
        values: `A`, `D`, `M`, `U` representing added, deleted, modified and same
        rows respectively. Note that if only one "status" is asked, by setting proper
        flags, this additional column is not created as it would have only one value
        for all rows. Beside additional diff column, new chain has schema of the chain
        on which method was called.

        Parameters:
            other: Chain to calculate diff from.
            on: Column or list of columns to match on. If both chains have the
                same columns then this column is enough for the match. Otherwise,
                `right_on` parameter has to specify the columns for the other chain.
                This value is used to find corresponding row in other dataset. If not
                found there, row is considered as added (or removed if vice versa), and
                if found then row can be either modified or same.
            right_on: Optional column or list of columns
                for the `other` to match.
            compare: Column or list of columns to compare on. If both chains have
                the same columns then this column is enough for the compare. Otherwise,
                `right_compare` parameter has to specify the columns for the other
                chain. This value is used to see if row is modified or same. If
                not set, all columns will be used for comparison
            right_compare: Optional column or list of columns
                    for the `other` to compare to.
            added (bool): Whether to return added rows in resulting chain.
            deleted (bool): Whether to return deleted rows in resulting chain.
            modified (bool): Whether to return modified rows in resulting chain.
            same (bool): Whether to return unchanged rows in resulting chain.
            status_col (str): Name of the new column that is created in resulting chain
                representing diff status.

        Example:
            ```py
            res = persons.compare(
                new_persons,
                on=["id"],
                right_on=["other_id"],
                compare=["name"],
                added=True,
                deleted=True,
                modified=True,
                same=True,
                status_col="diff"
            )
            ```
        """
        from datachain.diff import _compare

        return _compare(
            self,
            other,
            on,
            right_on=right_on,
            compare=compare,
            right_compare=right_compare,
            added=added,
            deleted=deleted,
            modified=modified,
            same=same,
            status_col=status_col,
        )

    def diff(
        self,
        other: "DataChain",
        on: str = "file",
        right_on: Optional[str] = None,
        added: bool = True,
        modified: bool = True,
        deleted: bool = False,
        same: bool = False,
        status_col: Optional[str] = None,
    ) -> "DataChain":
        """Similar to `.compare()`, which is more generic method to calculate difference
        between two chains. Unlike `.compare()`, this method works only on those chains
        that have `File` object, or it's derivatives, in it. File `source` and `path`
        are used for matching, and file `version` and `etag` for comparing, while in
        `.compare()` user needs to provide arbitrary columns for matching and comparing.

        Parameters:
            other: Chain to calculate diff from.
            on: File signal to match on. If both chains have the
                same file signal then this column is enough for the match. Otherwise,
                `right_on` parameter has to specify the file signal for the other chain.
                This value is used to find corresponding row in other dataset. If not
                found there, row is considered as added (or removed if vice versa), and
                if found then row can be either modified or same.
            right_on: Optional file signal for the `other` to match.
            added (bool): Whether to return added rows in resulting chain.
            deleted (bool): Whether to return deleted rows in resulting chain.
            modified (bool): Whether to return modified rows in resulting chain.
            same (bool): Whether to return unchanged rows in resulting chain.
            status_col (str): Optional name of the new column that is created in
                resulting chain representing diff status.

        Example:
            ```py
            diff = images.diff(
                new_images,
                on="file",
                right_on="other_file",
                added=True,
                deleted=True,
                modified=True,
                same=True,
                status_col="diff"
            )
            ```
        """
        on_file_signals = ["source", "path"]
        compare_file_signals = ["version", "etag"]

        def get_file_signals(file: str, signals):
            return [f"{file}.{c}" for c in signals]

        right_on = right_on or on

        on_cols = get_file_signals(on, on_file_signals)
        right_on_cols = get_file_signals(right_on, on_file_signals)
        compare_cols = get_file_signals(on, compare_file_signals)
        right_compare_cols = get_file_signals(right_on, compare_file_signals)

        return self.compare(
            other,
            on_cols,
            right_on=right_on_cols,
            compare=compare_cols,
            right_compare=right_compare_cols,
            added=added,
            deleted=deleted,
            modified=modified,
            same=same,
            status_col=status_col,
        )

    @classmethod
    def from_values(
        cls,
        ds_name: str = "",
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        output: OutputType = None,
        object_name: str = "",
        **fr_map,
    ) -> "Self":
        """Generate chain from list of values.

        Example:
            ```py
            DataChain.from_values(fib=[1, 2, 3, 5, 8])
            ```
        """
        tuple_type, output, tuples = values_to_tuples(ds_name, output, **fr_map)

        def _func_fr() -> Iterator[tuple_type]:  # type: ignore[valid-type]
            yield from tuples

        chain = cls.from_records(
            DataChain.DEFAULT_FILE_RECORD,
            session=session,
            settings=settings,
            in_memory=in_memory,
        )
        if object_name:
            output = {object_name: dict_to_data_model(object_name, output)}  # type: ignore[arg-type]
        return chain.gen(_func_fr, output=output)

    @classmethod
    def from_pandas(  # type: ignore[override]
        cls,
        df: "pd.DataFrame",
        name: str = "",
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        object_name: str = "",
    ) -> "DataChain":
        """Generate chain from pandas data-frame.

        Example:
            ```py
            import pandas as pd

            df = pd.DataFrame({"fib": [1, 2, 3, 5, 8]})
            DataChain.from_pandas(df)
            ```
        """
        fr_map = {col.lower(): df[col].tolist() for col in df.columns}

        for column in fr_map:
            if not column.isidentifier():
                raise DatasetPrepareError(
                    name,
                    f"import from pandas error - '{column}' cannot be a column name",
                )

        return cls.from_values(
            name,
            session,
            settings=settings,
            object_name=object_name,
            in_memory=in_memory,
            **fr_map,
        )

    def to_pandas(self, flatten=False, include_hidden=True) -> "pd.DataFrame":
        """Return a pandas DataFrame from the chain.

        Parameters:
            flatten : Whether to use a multiindex or flatten column names.
            include_hidden : Whether to include hidden columns.
        """
        import pandas as pd

        headers, max_length = self._effective_signals_schema.get_headers_with_length(
            include_hidden=include_hidden
        )
        if flatten or max_length < 2:
            columns = [".".join(filter(None, header)) for header in headers]
        else:
            columns = pd.MultiIndex.from_tuples(map(tuple, headers))

        results = self.results(include_hidden=include_hidden)
        return pd.DataFrame.from_records(results, columns=columns)

    def show(
        self,
        limit: int = 20,
        flatten=False,
        transpose=False,
        truncate=True,
        include_hidden=False,
    ) -> None:
        """Show a preview of the chain results.

        Parameters:
            limit : How many rows to show.
            flatten : Whether to use a multiindex or flatten column names.
            transpose : Whether to transpose rows and columns.
            truncate : Whether or not to truncate the contents of columns.
            include_hidden : Whether to include hidden columns.
        """
        import pandas as pd

        dc = self.limit(limit) if limit > 0 else self  # type: ignore[misc]
        df = dc.to_pandas(flatten, include_hidden=include_hidden)

        if df.empty:
            print("Empty result")
            print(f"Columns: {list(df.columns)}")
            return

        if transpose:
            df = df.T

        options: list = [
            "display.max_columns",
            None,
            "display.multi_sparse",
            False,
        ]

        try:
            if columns := os.get_terminal_size().columns:
                options.extend(["display.width", columns])
        except OSError:
            pass

        if not truncate:
            options.extend(["display.max_colwidth", None])

        with pd.option_context(*options):
            if inside_notebook():
                from IPython.display import display

                display(df)
            else:
                print(df)

        if len(df) == limit:
            print(f"\n[Limited by {len(df)} rows]")

    @classmethod
    def from_hf(
        cls,
        dataset: Union[str, "HFDatasetType"],
        *args,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        object_name: str = "",
        model_name: str = "",
        **kwargs,
    ) -> "DataChain":
        """Generate chain from huggingface hub dataset.

        Parameters:
            dataset : Path or name of the dataset to read from Hugging Face Hub,
                or an instance of `datasets.Dataset`-like object.
            session : Session to use for the chain.
            settings : Settings to use for the chain.
            object_name : Generated object column name.
            model_name : Generated model name.
            kwargs : Parameters to pass to datasets.load_dataset.

        Example:
            Load from Hugging Face Hub:
            ```py
            DataChain.from_hf("beans", split="train")
            ```

            Generate chain from loaded dataset:
            ```py
            from datasets import load_dataset
            ds = load_dataset("beans", split="train")
            DataChain.from_hf(ds)
            ```
        """
        from datachain.lib.hf import HFGenerator, get_output_schema, stream_splits

        output: dict[str, DataType] = {}
        ds_dict = stream_splits(dataset, *args, **kwargs)
        if len(ds_dict) > 1:
            output = {"split": str}

        model_name = model_name or object_name or ""
        hf_features = next(iter(ds_dict.values())).features
        output = output | get_output_schema(hf_features)
        model = dict_to_data_model(model_name, output)
        if object_name:
            output = {object_name: model}

        chain = DataChain.from_values(
            split=list(ds_dict.keys()), session=session, settings=settings
        )
        return chain.gen(HFGenerator(dataset, model, *args, **kwargs), output=output)

    def parse_tabular(
        self,
        output: OutputType = None,
        object_name: str = "",
        model_name: str = "",
        source: bool = True,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> "Self":
        """Generate chain from list of tabular files.

        Parameters:
            output : Dictionary or feature class defining column names and their
                corresponding types. List of column names is also accepted, in which
                case types will be inferred.
            object_name : Generated object column name.
            model_name : Generated model name.
            source : Whether to include info about the source file.
            nrows : Optional row limit.
            kwargs : Parameters to pass to pyarrow.dataset.dataset.

        Example:
            Reading a json lines file:
            ```py
            dc = DataChain.from_storage("s3://mybucket/file.jsonl")
            dc = dc.parse_tabular(format="json")
            ```

            Reading a filtered list of files as a dataset:
            ```py
            dc = DataChain.from_storage("s3://mybucket")
            dc = dc.filter(C("file.name").glob("*.jsonl"))
            dc = dc.parse_tabular(format="json")
            ```
        """
        from pyarrow.dataset import CsvFileFormat, JsonFileFormat

        from datachain.lib.arrow import ArrowGenerator, infer_schema, schema_to_output

        if nrows:
            format = kwargs.get("format")
            if format not in ["csv", "json"] and not isinstance(
                format, (CsvFileFormat, JsonFileFormat)
            ):
                raise DatasetPrepareError(
                    self.name,
                    "error in `parse_tabular` - "
                    "`nrows` only supported for csv and json formats.",
                )

        if "file" not in self.schema or not self.count():
            raise DatasetPrepareError(self.name, "no files to parse.")

        schema = None
        col_names = output if isinstance(output, Sequence) else None
        if col_names or not output:
            try:
                schema = infer_schema(self, **kwargs)
                output, _ = schema_to_output(schema, col_names)
            except ValueError as e:
                raise DatasetPrepareError(self.name, e) from e

        if isinstance(output, dict):
            model_name = model_name or object_name or ""
            model = dict_to_data_model(model_name, output)
            output = model
        else:
            model = output  # type: ignore[assignment]

        if object_name:
            output = {object_name: model}  # type: ignore[dict-item]
        elif isinstance(output, type(BaseModel)):
            output = {
                name: info.annotation  # type: ignore[misc]
                for name, info in output.model_fields.items()
            }

        if source:
            output = {"source": ArrowRow} | output  # type: ignore[assignment,operator]

        # disable prefetch if nrows is set
        settings = {"prefetch": 0} if nrows else {}
        return self.settings(**settings).gen(  # type: ignore[arg-type]
            ArrowGenerator(schema, model, source, nrows, **kwargs), output=output
        )

    @classmethod
    def from_csv(
        cls,
        path,
        delimiter: Optional[str] = None,
        header: bool = True,
        output: OutputType = None,
        object_name: str = "",
        model_name: str = "",
        source: bool = True,
        nrows=None,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        column_types: Optional[dict[str, "Union[str, ArrowDataType]"]] = None,
        parse_options: Optional[dict[str, "Union[str, Union[bool, Callable]]"]] = None,
        **kwargs,
    ) -> "DataChain":
        """Generate chain from csv files.

        Parameters:
            path : Storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///".
            delimiter : Character for delimiting columns. Takes precedence if also
                specified in `parse_options`. Defaults to ",".
            header : Whether the files include a header row.
            output : Dictionary or feature class defining column names and their
                corresponding types. List of column names is also accepted, in which
                case types will be inferred.
            object_name : Created object column name.
            model_name : Generated model name.
            source : Whether to include info about the source file.
            nrows : Optional row limit.
            session : Session to use for the chain.
            settings : Settings to use for the chain.
            column_types : Dictionary of column names and their corresponding types.
                It is passed to CSV reader and for each column specified type auto
                inference is disabled.
            parse_options: Tells the parser how to process lines.
                See https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html

        Example:
            Reading a csv file:
            ```py
            dc = DataChain.from_csv("s3://mybucket/file.csv")
            ```

            Reading csv files from a directory as a combined dataset:
            ```py
            dc = DataChain.from_csv("s3://mybucket/dir")
            ```
        """
        from pandas.io.parsers.readers import STR_NA_VALUES
        from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions
        from pyarrow.dataset import CsvFileFormat
        from pyarrow.lib import type_for_alias

        parse_options = parse_options or {}
        if "delimiter" not in parse_options:
            parse_options["delimiter"] = ","
        if delimiter:
            parse_options["delimiter"] = delimiter

        if column_types:
            column_types = {
                name: type_for_alias(typ) if isinstance(typ, str) else typ
                for name, typ in column_types.items()
            }
        else:
            column_types = {}

        chain = DataChain.from_storage(
            path, session=session, settings=settings, **kwargs
        )

        column_names = None
        if not header:
            if not output:
                msg = "error parsing csv - provide output if no header"
                raise DatasetPrepareError(chain.name, msg)
            if isinstance(output, Sequence):
                column_names = output  # type: ignore[assignment]
            elif isinstance(output, dict):
                column_names = list(output.keys())
            elif (fr := ModelStore.to_pydantic(output)) is not None:
                column_names = list(fr.model_fields.keys())
            else:
                msg = f"error parsing csv - incompatible output type {type(output)}"
                raise DatasetPrepareError(chain.name, msg)

        parse_options = ParseOptions(**parse_options)
        read_options = ReadOptions(column_names=column_names)
        convert_options = ConvertOptions(
            strings_can_be_null=True,
            null_values=STR_NA_VALUES,
            column_types=column_types,
        )
        format = CsvFileFormat(
            parse_options=parse_options,
            read_options=read_options,
            convert_options=convert_options,
        )
        return chain.parse_tabular(
            output=output,
            object_name=object_name,
            model_name=model_name,
            source=source,
            nrows=nrows,
            format=format,
        )

    @classmethod
    def from_parquet(
        cls,
        path,
        partitioning: Any = "hive",
        output: Optional[dict[str, DataType]] = None,
        object_name: str = "",
        model_name: str = "",
        source: bool = True,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        **kwargs,
    ) -> "DataChain":
        """Generate chain from parquet files.

        Parameters:
            path : Storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///".
            partitioning : Any pyarrow partitioning schema.
            output : Dictionary defining column names and their corresponding types.
            object_name : Created object column name.
            model_name : Generated model name.
            source : Whether to include info about the source file.
            session : Session to use for the chain.
            settings : Settings to use for the chain.

        Example:
            Reading a single file:
            ```py
            dc = DataChain.from_parquet("s3://mybucket/file.parquet")
            ```

            Reading a partitioned dataset from a directory:
            ```py
            dc = DataChain.from_parquet("s3://mybucket/dir")
            ```
        """
        chain = DataChain.from_storage(
            path, session=session, settings=settings, **kwargs
        )
        return chain.parse_tabular(
            output=output,
            object_name=object_name,
            model_name=model_name,
            source=source,
            format="parquet",
            partitioning=partitioning,
        )

    def to_parquet(
        self,
        path: Union[str, os.PathLike[str], BinaryIO],
        partition_cols: Optional[Sequence[str]] = None,
        chunk_size: int = DEFAULT_PARQUET_CHUNK_SIZE,
        fs_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Save chain to parquet file with SignalSchema metadata.

        Parameters:
            path : Path or a file-like binary object to save the file. This supports
                local paths as well as remote paths, such as s3:// or hf:// with fsspec.
            partition_cols : Column names by which to partition the dataset.
            chunk_size : The chunk size of results to read and convert to columnar
                data, to avoid running out of memory.
            fs_kwargs : Optional kwargs to pass to the fsspec filesystem, used only for
                write, for fsspec-type URLs, such as s3:// or hf:// when
                provided as the destination path.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from datachain.lib.arrow import DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY

        fsspec_fs = None

        if isinstance(path, str) and "://" in path:
            from datachain.client.fsspec import Client

            fs_kwargs = {
                **self._query.catalog.client_config,
                **(fs_kwargs or {}),
            }

            client = Client.get_implementation(path)

            if path.startswith("file://"):
                # pyarrow does not handle file:// uris, and needs a direct path instead.
                from urllib.parse import urlparse

                path = urlparse(path).path
                if sys.platform == "win32":
                    path = os.path.normpath(path.lstrip("/"))

            fsspec_fs = client.create_fs(**fs_kwargs)

        _partition_cols = list(partition_cols) if partition_cols else None
        signal_schema_metadata = orjson.dumps(
            self._effective_signals_schema.serialize()
        )

        column_names, column_chunks = self.to_columnar_data_with_names(chunk_size)

        parquet_schema = None
        parquet_writer = None
        first_chunk = True

        for chunk in column_chunks:
            # pyarrow infers the best parquet schema from the python types of
            # the input data.
            table = pa.Table.from_pydict(
                dict(zip(column_names, chunk)),
                schema=parquet_schema,
            )

            # Preserve any existing metadata, and add the DataChain SignalSchema.
            existing_metadata = table.schema.metadata or {}
            merged_metadata = {
                **existing_metadata,
                DATACHAIN_SIGNAL_SCHEMA_PARQUET_KEY: signal_schema_metadata,
            }
            table = table.replace_schema_metadata(merged_metadata)
            parquet_schema = table.schema

            if _partition_cols:
                # Write to a partitioned parquet dataset.
                pq.write_to_dataset(
                    table,
                    root_path=path,
                    partition_cols=_partition_cols,
                    filesystem=fsspec_fs,
                    **kwargs,
                )
            else:
                if first_chunk:
                    # Write to a single parquet file.
                    parquet_writer = pq.ParquetWriter(
                        path, parquet_schema, filesystem=fsspec_fs, **kwargs
                    )
                    first_chunk = False

                assert parquet_writer
                parquet_writer.write_table(table)

        if parquet_writer:
            parquet_writer.close()

    def to_csv(
        self,
        path: Union[str, os.PathLike[str]],
        delimiter: str = ",",
        fs_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Save chain to a csv (comma-separated values) file.

        Parameters:
            path : Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            delimiter : Delimiter to use for the resulting file.
            fs_kwargs : Optional kwargs to pass to the fsspec filesystem, used only for
                write, for fsspec-type URLs, such as s3:// or hf:// when
                provided as the destination path.
        """
        import csv

        opener = open

        if isinstance(path, str) and "://" in path:
            from datachain.client.fsspec import Client

            fs_kwargs = {
                **self._query.catalog.client_config,
                **(fs_kwargs or {}),
            }

            client = Client.get_implementation(path)

            fsspec_fs = client.create_fs(**fs_kwargs)

            opener = fsspec_fs.open

        headers, _ = self._effective_signals_schema.get_headers_with_length()
        column_names = [".".join(filter(None, header)) for header in headers]

        results_iter = self.collect_flatten()

        with opener(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter, **kwargs)
            writer.writerow(column_names)

            for row in results_iter:
                writer.writerow(row)

    def to_json(
        self,
        path: Union[str, os.PathLike[str]],
        fs_kwargs: Optional[dict[str, Any]] = None,
        include_outer_list: bool = True,
    ) -> None:
        """Save chain to a JSON file.

        Parameters:
            path : Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            fs_kwargs : Optional kwargs to pass to the fsspec filesystem, used only for
                write, for fsspec-type URLs, such as s3:// or hf:// when
                provided as the destination path.
            include_outer_list : Sets whether to include an outer list for all rows.
                Setting this to True makes the file valid JSON, while False instead
                writes in the JSON lines format.
        """
        opener = open

        if isinstance(path, str) and "://" in path:
            from datachain.client.fsspec import Client

            fs_kwargs = {
                **self._query.catalog.client_config,
                **(fs_kwargs or {}),
            }

            client = Client.get_implementation(path)

            fsspec_fs = client.create_fs(**fs_kwargs)

            opener = fsspec_fs.open

        headers, _ = self._effective_signals_schema.get_headers_with_length()
        headers = [list(filter(None, header)) for header in headers]

        is_first = True

        with opener(path, "wb") as f:
            if include_outer_list:
                # This makes the file JSON instead of JSON lines.
                f.write(b"[\n")
            for row in self.collect_flatten():
                if not is_first:
                    if include_outer_list:
                        # This makes the file JSON instead of JSON lines.
                        f.write(b",\n")
                    else:
                        f.write(b"\n")
                else:
                    is_first = False
                f.write(orjson.dumps(row_to_nested_dict(headers, row)))
            if include_outer_list:
                # This makes the file JSON instead of JSON lines.
                f.write(b"\n]\n")

    def to_jsonl(
        self,
        path: Union[str, os.PathLike[str]],
        fs_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save chain to a JSON lines file.

        Parameters:
            path : Path to save the file. This supports local paths as well as
                remote paths, such as s3:// or hf:// with fsspec.
            fs_kwargs : Optional kwargs to pass to the fsspec filesystem, used only for
                write, for fsspec-type URLs, such as s3:// or hf:// when
                provided as the destination path.
        """
        self.to_json(path, fs_kwargs, include_outer_list=False)

    @classmethod
    def from_records(
        cls,
        to_insert: Optional[Union[dict, list[dict]]],
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        schema: Optional[dict[str, DataType]] = None,
    ) -> "Self":
        """Create a DataChain from the provided records. This method can be used for
        programmatically generating a chain in contrast of reading data from storages
        or other sources.

        Parameters:
            to_insert : records (or a single record) to insert. Each record is
                        a dictionary of signals and theirs values.
            schema : describes chain signals and their corresponding types

        Example:
            ```py
            single_record = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD)
            ```
        """
        session = Session.get(session, in_memory=in_memory)
        catalog = session.catalog

        name = session.generate_temp_dataset_name()
        signal_schema = None
        columns: list[sqlalchemy.Column] = []

        if schema:
            signal_schema = SignalSchema(schema)
            columns = [
                sqlalchemy.Column(c.name, c.type)  # type: ignore[union-attr]
                for c in signal_schema.db_signals(as_columns=True)  # type: ignore[assignment]
            ]
        else:
            columns = [
                sqlalchemy.Column(name, typ)
                for name, typ in File._datachain_column_types.items()
            ]

        dsr = catalog.create_dataset(
            name,
            columns=columns,
            feature_schema=(
                signal_schema.clone_without_sys_signals().serialize()
                if signal_schema
                else None
            ),
        )

        session.add_dataset_version(dsr, dsr.latest_version)

        if isinstance(to_insert, dict):
            to_insert = [to_insert]
        elif not to_insert:
            to_insert = []

        warehouse = catalog.warehouse
        dr = warehouse.dataset_rows(dsr)
        db = warehouse.db
        insert_q = dr.get_table().insert()
        for record in to_insert:
            db.execute(insert_q.values(**record))
        return cls.from_dataset(name=dsr.name, session=session, settings=settings)

    def sum(self, fr: DataType):  # type: ignore[override]
        """Compute the sum of a column."""
        return self._extend_to_data_model("sum", fr)

    def avg(self, fr: DataType):  # type: ignore[override]
        """Compute the average of a column."""
        return self._extend_to_data_model("avg", fr)

    def min(self, fr: DataType):  # type: ignore[override]
        """Compute the minimum of a column."""
        return self._extend_to_data_model("min", fr)

    def max(self, fr: DataType):  # type: ignore[override]
        """Compute the maximum of a column."""
        return self._extend_to_data_model("max", fr)

    def setup(self, **kwargs) -> "Self":
        """Setup variables to pass to UDF functions.

        Use before running map/gen/agg/batch_map to save an object and pass it as an
        argument to the UDF.

        Example:
            ```py
            import anthropic
            from anthropic.types import Message

            (
                DataChain.from_storage(DATA, type="text")
                .settings(parallel=4, cache=True)
                .setup(client=lambda: anthropic.Anthropic(api_key=API_KEY))
                .map(
                    claude=lambda client, file: client.messages.create(
                        model=MODEL,
                        system=PROMPT,
                        messages=[{"role": "user", "content": file.get_value()}],
                    ),
                    output=Message,
                )
            )
            ```
        """
        intersection = set(self._setup.keys()) & set(kwargs.keys())
        if intersection:
            keys = ", ".join(intersection)
            raise DatasetPrepareError(self.name, f"this value(s) already setup: {keys}")

        self._setup = self._setup | kwargs
        return self

    def to_storage(
        self,
        output: Union[str, os.PathLike[str]],
        signal: str = "file",
        placement: FileExportPlacement = "fullpath",
        link_type: Literal["copy", "symlink"] = "copy",
        num_threads: Optional[int] = EXPORT_FILES_MAX_THREADS,
        anon: bool = False,
        client_config: Optional[dict] = None,
    ) -> None:
        """Export files from a specified signal to a directory. Files can be
        exported to a local or cloud directory.

        Args:
            output: Path to the target directory for exporting files.
            signal: Name of the signal to export files from.
            placement: The method to use for naming exported files.
                The possible values are: "filename", "etag", "fullpath", and "checksum".
            link_type: Method to use for exporting files.
                Falls back to `'copy'` if symlinking fails.
            num_threads : number of threads to use for exporting files.
                By default it uses 5 threads.
            anon: If true, we will treat cloud bucket as public one
            client_config: Optional configuration for the destination storage client

        Example:
            Cross cloud transfer
            ```py
            ds = DataChain.from_storage("s3://mybucket")
            ds.to_storage("gs://mybucket", placement="filename")
            ```
        """
        if placement == "filename" and (
            self._query.distinct(pathfunc.name(C(f"{signal}__path"))).count()
            != self._query.count()
        ):
            raise ValueError("Files with the same name found")

        if anon:
            client_config = (client_config or {}) | {"anon": True}

        progress_bar = tqdm(
            desc=f"Exporting files to {output}: ",
            unit=" files",
            unit_scale=True,
            unit_divisor=10,
            total=self.count(),
            leave=False,
        )
        file_exporter = FileExporter(
            output,
            placement,
            self._settings.cache if self._settings else False,
            link_type,
            max_threads=num_threads or 1,
            client_config=client_config,
        )
        file_exporter.run(self.collect(signal), progress_bar)

    def shuffle(self) -> "Self":
        """Shuffle the rows of the chain deterministically."""
        return self.order_by("sys.rand")

    def sample(self, n) -> "Self":
        """Return a random sample from the chain.

        Parameters:
            n (int): Number of samples to draw.

        NOTE: Samples are not deterministic, and streamed/paginated queries or
        multiple workers will draw samples with replacement.
        """
        return self._evolve(query=self._query.sample(n))

    def filter(self, *args: Any) -> "Self":
        """Filter the chain according to conditions.

        Example:
            Basic usage with built-in operators
            ```py
            dc.filter(C("width") < 200)
            ```

            Using glob to match patterns
            ```py
            dc.filter(C("file.name").glob("*.jpg"))
            ```

            Using `datachain.func`
            ```py
            from datachain.func import string
            dc.filter(string.length(C("file.name")) > 5)
            ```

            Combining filters with "or"
            ```py
            dc.filter(C("file.name").glob("cat*") | C("file.name").glob("dog*))
            ```

            Combining filters with "and"
            ```py
            dc.filter(
                C("file.name").glob("*.jpg) &
                (string.length(C("file.name")) > 5)
            )
            ```
        """
        return self._evolve(query=self._query.filter(*args))

    def limit(self, n: int) -> "Self":
        """Return the first `n` rows of the chain.

        If the chain is unordered, which rows are returned is undefined.
        If the chain has less than `n` rows, the whole chain is returned.

        Parameters:
            n (int): Number of rows to return.
        """
        return self._evolve(query=self._query.limit(n))

    def offset(self, offset: int) -> "Self":
        """Return the results starting with the offset row.

        If the chain is unordered, which rows are skipped in undefined.
        If the chain has less than `offset` rows, the result is an empty chain.

        Parameters:
            offset (int): Number of rows to skip.
        """
        return self._evolve(query=self._query.offset(offset))

    def count(self) -> int:
        """Return the number of rows in the chain."""
        return self._query.count()

    def exec(self) -> "Self":
        """Execute the chain."""
        return self._evolve(query=self._query.exec())

    def chunk(self, index: int, total: int) -> "Self":
        """Split a chain into smaller chunks for e.g. parallelization.

        Example:
            ```py
            chain = DataChain.from_storage(...)
            chunk_1 = query._chunk(0, 2)
            chunk_2 = query._chunk(1, 2)
            ```

        Note:
            Bear in mind that `index` is 0-indexed but `total` isn't.
            Use 0/3, 1/3 and 2/3, not 1/3, 2/3 and 3/3.
        """
        return self._evolve(query=self._query.chunk(index, total))
