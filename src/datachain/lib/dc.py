import copy
import os
import re
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

import pandas as pd
import sqlalchemy
from pydantic import BaseModel
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.sql.sqltypes import NullType

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.values_to_tuples import values_to_tuples
from datachain.lib.data_model import DataModel, DataType, dict_to_data_model
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.file import ExportPlacement as FileExportPlacement
from datachain.lib.file import File, IndexedFile, get_file
from datachain.lib.meta_formats import read_meta, read_schema
from datachain.lib.model_store import ModelStore
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import (
    Aggregator,
    BatchMapper,
    Generator,
    Mapper,
    UDFBase,
)
from datachain.lib.udf_signature import UdfSignature
from datachain.lib.utils import DataChainParamsError
from datachain.query import Session
from datachain.query.dataset import (
    DatasetQuery,
    PartitionByType,
    detach,
)
from datachain.query.schema import Column, DatasetRow
from datachain.sql.functions import path as pathfunc
from datachain.utils import inside_notebook

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec, Self

    from datachain.lib.hf import HFDatasetType

    P = ParamSpec("P")

C = Column

_T = TypeVar("_T")
D = TypeVar("D", bound="DataChain")


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


class DatasetMergeError(DataChainParamsError):  # noqa: D101
    def __init__(self, on: Sequence[str], right_on: Optional[Sequence[str]], msg: str):  # noqa: D107
        on_str = ", ".join(on) if isinstance(on, Sequence) else ""
        right_on_str = (
            ", right_on='" + ", ".join(right_on) + "'"
            if right_on and isinstance(right_on, Sequence)
            else ""
        )
        super().__init__(f"Merge error on='{on_str}'{right_on_str}: {msg}")


class DataChainColumnError(DataChainParamsError):  # noqa: D101
    def __init__(self, col_name, msg):  # noqa: D107
        super().__init__(f"Error for column {col_name}: {msg}")


OutputType = Union[None, DataType, Sequence[str], dict[str, DataType]]


class Sys(DataModel):
    """Model for internal DataChain signals `id` and `rand`."""

    id: int
    rand: int


class DataChain(DatasetQuery):
    """AI 🔗 DataChain - a data structure for batch data processing and evaluation.

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

        from datachain.dc import DataChain, Column

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
        "vtype": "",
        "size": 0,
    }

    def __init__(self, *args, settings: Optional[dict] = None, **kwargs):
        """This method needs to be redefined as a part of Dataset and DataChain
        decoupling.
        """
        super().__init__(  # type: ignore[misc]
            *args,
            **kwargs,
            indexing_column_types=File._datachain_column_types,
        )
        if settings:
            self._settings = Settings(**settings)
        else:
            self._settings = Settings()
        self._setup: dict = {}

        self.signals_schema = SignalSchema({"sys": Sys})
        if self.feature_schema:
            self.signals_schema |= SignalSchema.deserialize(self.feature_schema)
        else:
            self.signals_schema |= SignalSchema.from_column_types(
                self.column_types or {}
            )

        self._sys = False

    @property
    def schema(self) -> dict[str, DataType]:
        """Get schema of the chain."""
        return self._effective_signals_schema.values

    def column(self, name: str) -> Column:
        """Returns Column instance with a type if name is found in current schema,
        otherwise raises an exception.
        """
        name_path = name.split(".")
        for path, type_, _, _ in self.signals_schema.get_flat_tree():
            if path == name_path:
                return Column(name, python_to_sql(type_))

        raise ValueError(f"Column with name {name} not found in the schema")

    def print_schema(self) -> None:
        """Print schema of the chain."""
        self._effective_signals_schema.print_tree()

    def clone(self, new_table: bool = True) -> "Self":
        """Make a copy of the chain in a new table."""
        obj = super().clone(new_table=new_table)
        obj.signals_schema = copy.deepcopy(self.signals_schema)
        return obj

    def settings(
        self,
        cache=None,
        parallel=None,
        workers=None,
        min_task_size=None,
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

        Example:
            ```py
            chain = (
                chain
                .settings(cache=True, parallel=8)
                .map(laion=process_webdataset(spec=WDSLaion), params="file")
            )
            ```
        """
        chain = self.clone()
        if sys is not None:
            chain._sys = sys
        chain._settings.add(Settings(cache, parallel, workers, min_task_size))
        return chain

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
        path,
        *,
        type: Literal["binary", "text", "image"] = "binary",
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        recursive: Optional[bool] = True,
        object_name: str = "file",
        update: bool = False,
        **kwargs,
    ) -> "Self":
        """Get data from a storage as a list of file with all file attributes.
        It returns the chain itself as usual.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            recursive : search recursively for the given path.
            object_name : Created object column name.
            update : force storage reindexing. Default is False.

        Example:
            ```py
            chain = DataChain.from_storage("s3://my-bucket/my-dir")
            ```
        """
        func = get_file(type)
        return (
            cls(
                path,
                session=session,
                settings=settings,
                recursive=recursive,
                update=update,
                in_memory=in_memory,
                **kwargs,
            )
            .map(**{object_name: func})
            .select(object_name)
        )

    @classmethod
    def from_dataset(cls, name: str, version: Optional[int] = None) -> "DataChain":
        """Get data from a saved Dataset. It returns the chain itself.

        Parameters:
            name : dataset name
            version : dataset version

        Example:
            ```py
            chain = DataChain.from_dataset("my_cats")
            ```
        """
        return DataChain(name=name, version=version)

    @classmethod
    def from_json(
        cls,
        path,
        type: Literal["binary", "text", "image"] = "text",
        spec: Optional[DataType] = None,
        schema_from: Optional[str] = "auto",
        jmespath: Optional[str] = None,
        object_name: Optional[str] = "",
        model_name: Optional[str] = None,
        print_schema: Optional[bool] = False,
        meta_type: Optional[str] = "json",
        nrows=None,
        **kwargs,
    ) -> "DataChain":
        """Get data from JSON. It returns the chain itself.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            spec : optional Data Model
            schema_from : path to sample to infer spec (if schema not provided)
            object_name : generated object column name
            model_name : optional generated model name
            print_schema : print auto-generated schema
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
            schema_from = path

        def jmespath_to_name(s: str):
            name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
            return s[:name_end]

        if (not object_name) and jmespath:
            object_name = jmespath_to_name(jmespath)
        if not object_name:
            object_name = meta_type
        chain = DataChain.from_storage(path=path, type=type, **kwargs)
        signal_dict = {
            object_name: read_meta(
                schema_from=schema_from,
                meta_type=meta_type,
                spec=spec,
                model_name=model_name,
                print_schema=print_schema,
                jmespath=jmespath,
                nrows=nrows,
            )
        }
        return chain.gen(**signal_dict)  # type: ignore[misc, arg-type]

    @classmethod
    def from_jsonl(
        cls,
        path,
        type: Literal["binary", "text", "image"] = "text",
        spec: Optional[DataType] = None,
        schema_from: Optional[str] = "auto",
        jmespath: Optional[str] = None,
        object_name: Optional[str] = "",
        model_name: Optional[str] = None,
        print_schema: Optional[bool] = False,
        meta_type: Optional[str] = "jsonl",
        nrows=None,
        **kwargs,
    ) -> "DataChain":
        """Get data from JSON lines. It returns the chain itself.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            spec : optional Data Model
            schema_from : path to sample to infer spec (if schema not provided)
            object_name : generated object column name
            model_name : optional generated model name
            print_schema : print auto-generated schema
            jmespath : optional JMESPATH expression to reduce JSON
            nrows : optional row limit for jsonl and JSON arrays

        Example:
            infer JSONl schema from data, limit parsing to 1 row
            ```py
            chain = DataChain.from_jsonl("gs://myjsonl", nrows=1)
            ```
        """
        if schema_from == "auto":
            schema_from = path

        def jmespath_to_name(s: str):
            name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
            return s[:name_end]

        if (not object_name) and jmespath:
            object_name = jmespath_to_name(jmespath)
        if not object_name:
            object_name = meta_type
        chain = DataChain.from_storage(path=path, type=type, **kwargs)
        signal_dict = {
            object_name: read_meta(
                schema_from=schema_from,
                meta_type=meta_type,
                spec=spec,
                model_name=model_name,
                print_schema=print_schema,
                jmespath=jmespath,
                nrows=nrows,
            )
        }
        return chain.gen(**signal_dict)  # type: ignore[misc, arg-type]

    @classmethod
    def datasets(
        cls,
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        object_name: str = "dataset",
    ) -> "DataChain":
        """Generate chain with list of registered datasets.

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
            for d, v, j in catalog.list_datasets_versions()
        ]

        return cls.from_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            output={object_name: DatasetInfo},
            **{object_name: datasets},  # type: ignore[arg-type]
        )

    def print_json_schema(  # type: ignore[override]
        self, jmespath: Optional[str] = None, model_name: Optional[str] = None
    ) -> "Self":
        """Print JSON data model and save it. It returns the chain itself.

        Parameters:
            jmespath : JMESPATH expression to reduce JSON
            model_name : generated model name

        Example:
            print JSON schema and save to column "meta_from":
            ```py
            uri = "gs://datachain-demo/coco2017/annotations_captions/"
            chain = DataChain.from_storage(uri)
            chain = chain.show_json_schema()
            chain.save()
            ```
        """
        return self.map(
            meta_schema=lambda file: read_schema(
                file, data_type="json", expr=jmespath, model_name=model_name
            ),
            output=str,
        )

    def print_jsonl_schema(  # type: ignore[override]
        self, jmespath: Optional[str] = None, model_name: Optional[str] = None
    ) -> "Self":
        """Print JSON data model and save it. It returns the chain itself.

        Parameters:
            jmespath : JMESPATH expression to reduce JSON
            model_name : generated model name
        """
        return self.map(
            meta_schema=lambda file: read_schema(
                file, data_type="jsonl", expr=jmespath, model_name=model_name
            ),
            output=str,
        )

    def save(  # type: ignore[override]
        self, name: Optional[str] = None, version: Optional[int] = None
    ) -> "Self":
        """Save to a Dataset. It returns the chain itself.

        Parameters:
            name : dataset name. Empty name saves to a temporary dataset that will be
                removed after process ends. Temp dataset are useful for optimization.
            version : version of a dataset. Default - the last version that exist.
        """
        schema = self.signals_schema.clone_without_sys_signals().serialize()
        return super().save(name=name, version=version, feature_schema=schema)

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

        chain = self.add_signals(
            udf_obj.to_udf_wrapper(),
            **self._settings.to_dict(),
        )

        return chain.add_schema(udf_obj.output).reset_settings(self._settings)

    def gen(
        self,
        func: Optional[Callable] = None,
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
        chain = DatasetQuery.generate(
            self,
            udf_obj.to_udf_wrapper(),
            **self._settings.to_dict(),
        )

        return chain.reset_schema(udf_obj.output).reset_settings(self._settings)

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

        Example:
            ```py
            chain = chain.agg(
                total=lambda category, amount: [sum(amount)],
                output=float,
                partition_by="category",
            )
            chain.save("new_dataset")
            ```
        """
        udf_obj = self._udf_to_obj(Aggregator, func, params, output, signal_map)
        chain = DatasetQuery.generate(
            self,
            udf_obj.to_udf_wrapper(),
            partition_by=partition_by,
            **self._settings.to_dict(),
        )

        return chain.reset_schema(udf_obj.output).reset_settings(self._settings)

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
        chain = DatasetQuery.add_signals(
            self,
            udf_obj.to_udf_wrapper(batch),
            **self._settings.to_dict(),
        )

        return chain.add_schema(udf_obj.output).reset_settings(self._settings)

    def _udf_to_obj(
        self,
        target_class: type[UDFBase],
        func: Optional[Callable],
        params: Union[None, str, Sequence[str]],
        output: OutputType,
        signal_map,
    ) -> UDFBase:
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
        super_func = getattr(super(), method_name)

        new_schema = self.signals_schema.resolve(*args)
        columns = [C(col) for col in new_schema.db_signals()]
        res = super_func(*columns, **kwargs)
        if isinstance(res, DataChain):
            res.signals_schema = new_schema

        return res

    @detach
    @resolve_columns
    def order_by(self, *args, descending: bool = False) -> "Self":
        """Orders by specified set of signals.

        Parameters:
            descending (bool): Whether to sort in descending order or not.
        """
        if descending:
            args = tuple(sqlalchemy.desc(a) for a in args)

        return super().order_by(*args)

    @detach
    def distinct(self, arg: str, *args: str) -> "Self":  # type: ignore[override]
        """Removes duplicate rows based on uniqueness of some input column(s)
        i.e if rows are found with the same value of input column(s), only one
        row is left in the result set.

        Example:
        ```py
         dc.distinct("file.parent", "file.name")
        )
        ```
        """
        return super().distinct(*self.signals_schema.resolve(arg, *args).db_signals())

    @detach
    def select(self, *args: str, _sys: bool = True) -> "Self":
        """Select only a specified set of signals."""
        new_schema = self.signals_schema.resolve(*args)
        if _sys:
            new_schema = SignalSchema({"sys": Sys}) | new_schema
        columns = new_schema.db_signals()
        chain = super().select(*columns)
        chain.signals_schema = new_schema
        return chain

    @detach
    def select_except(self, *args: str) -> "Self":
        """Select all the signals expect the specified signals."""
        new_schema = self.signals_schema.select_except_signals(*args)
        columns = new_schema.db_signals()
        chain = super().select(*columns)
        chain.signals_schema = new_schema
        return chain

    @detach
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
           String:      length(), split()
           Filename:    name(), parent(), file_stem(), file_ext()
           Array:       length(), sip_hash_64(), euclidean_distance(),
                        cosine_distance()

        Example:
        ```py
         dc.mutate(
                area=Column("image.height") * Column("image.width"),
                extension=file_ext(Column("file.name")),
                dist=cosine_distance(embedding_text, embedding_image)
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
        existing_columns = set(self.signals_schema.values.keys())
        for col_name in kwargs:
            if col_name in existing_columns:
                raise DataChainColumnError(
                    col_name,
                    "Cannot modify existing column with mutate(). "
                    "Use a different name for the new column.",
                )
        for col_name, expr in kwargs.items():
            if not isinstance(expr, Column) and isinstance(expr.type, NullType):
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
            else:
                # adding new signal
                mutated[name] = value

        chain = super().mutate(**mutated)
        chain.signals_schema = schema.mutate(kwargs)
        return chain

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
    def collect_flatten(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> Iterator[_T]: ...

    def collect_flatten(self, *, row_factory=None):
        """Yields flattened rows of values as a tuple.

        Args:
            row_factory : A callable to convert row to a custom format.
                          It should accept two arguments: a list of column names and
                          a tuple of row values.
        """
        db_signals = self._effective_signals_schema.db_signals()
        with super().select(*db_signals).as_iterable() as rows:
            if row_factory:
                rows = (row_factory(db_signals, r) for r in rows)
            yield from rows

    @overload
    def results(self) -> list[tuple[Any, ...]]: ...

    @overload
    def results(
        self, *, row_factory: Callable[[list[str], tuple[Any, ...]], _T]
    ) -> list[_T]: ...

    def results(self, *, row_factory=None):  # noqa: D102
        if row_factory is None:
            return list(self.collect_flatten())
        return list(self.collect_flatten(row_factory=row_factory))

    def to_records(self) -> list[dict[str, Any]]:
        """Convert every row to a dictionary."""

        def to_dict(cols: list[str], row: tuple[Any, ...]) -> dict[str, Any]:
            return dict(zip(cols, row))

        return self.results(row_factory=to_dict)

    @overload
    def collect(self) -> Iterator[tuple[DataType, ...]]: ...

    @overload
    def collect(self, col: str) -> Iterator[DataType]: ...  # type: ignore[overload-overlap]

    @overload
    def collect(self, *cols: str) -> Iterator[tuple[DataType, ...]]: ...

    def collect(self, *cols: str) -> Iterator[Union[DataType, tuple[DataType, ...]]]:  # type: ignore[overload-overlap,misc]
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
        with super().select(*db_signals).as_iterable() as rows:
            for row in rows:
                ret = signals_schema.row_to_features(
                    row, catalog=chain.session.catalog, cache=chain._settings.cache
                )
                yield ret[0] if len(cols) == 1 else tuple(ret)

    def to_pytorch(
        self, transform=None, tokenizer=None, tokenizer_kwargs=None, num_samples=0
    ):
        """Convert to pytorch dataset format.

        Args:
            transform (Transform): Torchvision transforms to apply to the dataset.
            tokenizer (Callable): Tokenizer to use to tokenize text values.
            tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
            num_samples (int): Number of random samples to draw for each epoch.
                This argument is ignored if `num_samples=0` (the default).

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

        if self.attached:
            chain = self
        else:
            chain = self.save()
        assert chain.name is not None  # for mypy
        return PytorchDataset(
            chain.name,
            chain.version,
            catalog=self.catalog,
            transform=transform,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            num_samples=num_samples,
        )

    def remove_file_signals(self) -> "Self":  # noqa: D102
        schema = self.signals_schema.clone_without_file_signals()
        return self.select(*schema.values.keys())

    @detach
    def merge(
        self,
        right_ds: "DataChain",
        on: Union[str, Sequence[str]],
        right_on: Union[str, Sequence[str], None] = None,
        inner=False,
        rname="right_",
    ) -> "Self":
        """Merge two chains based on the specified criteria.

        Parameters:
            right_ds : Chain to join with.
            on : Predicate or list of Predicates to join on. If both chains have the
                same predicates then this predicate is enough for the join. Otherwise,
                `right_on` parameter has to specify the predicates for the other chain.
            right_on: Optional predicate or list of Predicates
                    for the `right_ds` to join.
            inner (bool): Whether to run inner join or outer join.
            rname (str): name prefix for conflicting signal names.

        Example:
            ```py
            meta = meta_emd.merge(meta_pq, on=(C.name, C.emd__index),
                                  right_on=(C.name, C.pq__index))
            ```
        """
        if on is None:
            raise DatasetMergeError(["None"], None, "'on' must be specified")

        if isinstance(on, str):
            on = [on]
        elif not isinstance(on, Sequence):
            raise DatasetMergeError(
                on,
                right_on,
                f"'on' must be 'str' or 'Sequence' object but got type '{type(on)}'",
            )

        signals_schema = self.signals_schema.clone_without_sys_signals()
        on_columns: list[str] = signals_schema.resolve(*on).db_signals()  # type: ignore[assignment]

        right_signals_schema = right_ds.signals_schema.clone_without_sys_signals()
        if right_on is not None:
            if isinstance(right_on, str):
                right_on = [right_on]
            elif not isinstance(right_on, Sequence):
                raise DatasetMergeError(
                    on,
                    right_on,
                    "'right_on' must be 'str' or 'Sequence' object"
                    f" but got type '{right_on}'",
                )

            if len(right_on) != len(on):
                raise DatasetMergeError(
                    on, right_on, "'on' and 'right_on' must have the same length'"
                )

            right_on_columns: list[str] = right_signals_schema.resolve(
                *right_on
            ).db_signals()  # type: ignore[assignment]

            if len(right_on_columns) != len(on_columns):
                on_str = ", ".join(right_on_columns)
                right_on_str = ", ".join(right_on_columns)
                raise DatasetMergeError(
                    on,
                    right_on,
                    "'on' and 'right_on' must have the same number of columns in db'."
                    f" on -> {on_str}, right_on -> {right_on_str}",
                )
        else:
            right_on = on
            right_on_columns = on_columns

        if self == right_ds:
            right_ds = right_ds.clone(new_table=True)

        ops = [
            self.c(left) == right_ds.c(right)
            for left, right in zip(on_columns, right_on_columns)
        ]

        ds = self.join(right_ds, sqlalchemy.and_(*ops), inner, rname + "{name}")

        ds.feature_schema = None
        ds.signals_schema = SignalSchema({"sys": Sys}) | signals_schema.merge(
            right_signals_schema, rname
        )

        return ds

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

        return super()._subtract(other, signals)  # type: ignore[arg-type]

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
    ) -> "DataChain":
        """Generate chain from list of values.

        Example:
            ```py
            DataChain.from_values(fib=[1, 2, 3, 5, 8])
            ```
        """
        tuple_type, output, tuples = values_to_tuples(ds_name, output, **fr_map)

        def _func_fr() -> Iterator[tuple_type]:  # type: ignore[valid-type]
            yield from tuples

        chain = DataChain.from_records(
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
            if column in DatasetRow.schema:
                raise DatasetPrepareError(
                    name,
                    f"import from pandas error - column '{column}' conflicts with"
                    " default schema",
                )
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

    def to_pandas(self, flatten=False) -> "pd.DataFrame":
        """Return a pandas DataFrame from the chain.

        Parameters:
            flatten : Whether to use a multiindex or flatten column names.
        """
        headers, max_length = self._effective_signals_schema.get_headers_with_length()
        if flatten or max_length < 2:
            columns = [".".join(filter(None, header)) for header in headers]
        else:
            columns = pd.MultiIndex.from_tuples(map(tuple, headers))

        return pd.DataFrame.from_records(self.results(), columns=columns)

    def show(
        self,
        limit: int = 20,
        flatten=False,
        transpose=False,
        truncate=True,
    ) -> None:
        """Show a preview of the chain results.

        Parameters:
            limit : How many rows to show.
            flatten : Whether to use a multiindex or flatten column names.
            transpose : Whether to transpose rows and columns.
            truncate : Whether or not to truncate the contents of columns.
        """
        dc = self.limit(limit) if limit > 0 else self
        df = dc.to_pandas(flatten)

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
        output = output | get_output_schema(next(iter(ds_dict.values())), model_name)
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

        schema = None
        col_names = output if isinstance(output, Sequence) else None
        if col_names or not output:
            try:
                schema = infer_schema(self, **kwargs)
                output = schema_to_output(schema, col_names)
            except ValueError as e:
                raise DatasetPrepareError(self.name, e) from e

        if isinstance(output, dict):
            model_name = model_name or object_name or ""
            model = dict_to_data_model(model_name, output)
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
            output = {"source": IndexedFile} | output  # type: ignore[assignment,operator]
        return self.gen(
            ArrowGenerator(schema, model, source, nrows, **kwargs), output=output
        )

    @classmethod
    def from_csv(
        cls,
        path,
        delimiter: str = ",",
        header: bool = True,
        output: OutputType = None,
        object_name: str = "",
        model_name: str = "",
        source: bool = True,
        nrows=None,
        **kwargs,
    ) -> "DataChain":
        """Generate chain from csv files.

        Parameters:
            path : Storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///".
            delimiter : Character for delimiting columns.
            header : Whether the files include a header row.
            output : Dictionary or feature class defining column names and their
                corresponding types. List of column names is also accepted, in which
                case types will be inferred.
            object_name : Created object column name.
            model_name : Generated model name.
            source : Whether to include info about the source file.
            nrows : Optional row limit.

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

        chain = DataChain.from_storage(path, **kwargs)

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
        elif nrows:
            nrows += 1

        parse_options = ParseOptions(delimiter=delimiter)
        read_options = ReadOptions(column_names=column_names)
        convert_options = ConvertOptions(
            strings_can_be_null=True, null_values=STR_NA_VALUES
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
        chain = DataChain.from_storage(path, **kwargs)
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
        **kwargs,
    ) -> None:
        """Save chain to parquet file.

        Parameters:
            path : Path or a file-like binary object to save the file.
            partition_cols : Column names by which to partition the dataset.
        """
        _partition_cols = list(partition_cols) if partition_cols else None
        return self.to_pandas().to_parquet(
            path,
            partition_cols=_partition_cols,
            **kwargs,
        )

    @classmethod
    def from_records(
        cls,
        to_insert: Optional[Union[dict, list[dict]]],
        session: Optional[Session] = None,
        settings: Optional[dict] = None,
        in_memory: bool = False,
        schema: Optional[dict[str, DataType]] = None,
    ) -> "DataChain":
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
            columns = signal_schema.db_signals(as_columns=True)  # type: ignore[assignment]
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
        return DataChain(name=dsr.name, settings=settings)

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

    def export_files(
        self,
        output: str,
        signal="file",
        placement: FileExportPlacement = "fullpath",
        use_cache: bool = True,
    ) -> None:
        """Method that exports all files from chain to some folder."""
        if placement == "filename" and (
            super().distinct(pathfunc.name(C(f"{signal}__path"))).count()
            != self.count()
        ):
            raise ValueError("Files with the same name found")

        for file in self.collect(signal):
            file.export(output, placement, use_cache)  # type: ignore[union-attr]

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
        return super().sample(n)

    @detach
    def filter(self, *args) -> "Self":
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

            Using `datachain.sql.functions`
            ```py
            from datachain.sql.functions import string
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
        return super().filter(*args)

    @detach
    def limit(self, n: int) -> "Self":
        """Return the first n rows of the chain."""
        return super().limit(n)

    @detach
    def offset(self, offset: int) -> "Self":
        """Return the results starting with the offset row."""
        return super().offset(offset)
