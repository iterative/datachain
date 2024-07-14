import re
from collections.abc import Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Union,
)

import sqlalchemy

from datachain.lib.feature import FeatureType
from datachain.lib.feature_utils import features_to_tuples
from datachain.lib.file import File, IndexedFile, get_file
from datachain.lib.meta_formats import read_meta, read_schema
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

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import Self

    from datachain.catalog import Catalog

C = Column


class DatasetPrepareError(DataChainParamsError):
    def __init__(self, name, msg, output=None):
        name = f" '{name}'" if name else ""
        output = f" output '{output}'" if output else ""
        super().__init__(f"Dataset{name}{output} processing prepare error: {msg}")


class DatasetFromFeatureError(DataChainParamsError):
    def __init__(self, name, msg):
        name = f" '{name}'" if name else ""
        super().__init__(f"Dataset {name} from feature error: {msg}")


class DatasetMergeError(DataChainParamsError):
    def __init__(self, on: Sequence[str], right_on: Optional[Sequence[str]], msg: str):
        on_str = ", ".join(on) if isinstance(on, Sequence) else ""
        right_on_str = (
            ", right_on='" + ", ".join(right_on) + "'"
            if right_on and isinstance(right_on, Sequence)
            else ""
        )
        super().__init__(f"Merge error on='{on_str}'{right_on_str}: {msg}")


class DataChain(DatasetQuery):
    """AI 🔗 DataChain - a data structure for batch data processing and evaluation.

    It represents a sequence of data manipulation steps such as reading data from
    storages, running AI or LLM models or calling external services API to validate or
    enrich data.

    Data in DataChain is presented as Python classes with arbitrary set of fields,
    including nested classes. The data classes have to inherit from `Feature` class.
    The supported set of field types include: majority of the type supported by the
    underlyind library `Pydantic`.

    See Also:
        `DataChain.from_storage("s3://my-bucket/my-dir/")` - reading unstructured
            data files from storages such as S3, gs or Azure ADLS.

        `DataChain.save("name")` - saving to a dataset.

        `DataChain.from_dataset("name")` - reading from a dataset.

        `DataChain.from_features(fib=[1, 2, 3, 5, 8])` - generating from a values.


    Example:
        ```py
        from datachain import DataChain, Feature
        from datachain.lib.claude import claude_processor

        class Rating(Feature):
        status: str = ""
        explanation: str = ""

        PROMPT = "A 'user' is a human trying to find the best mobile plan.... "
        MODEL = "claude-3-opus-20240229"

        chain = (
            DataChain.from_storage("s3://my-bucket/my")
            .filter(C.name.glob("*.txt"))
            .limit(5)
            .map(claude=claude_processor(prompt=PROMPT, model=MODEL))
            .map(
                rating=lambda claude: Rating(
                    **(json.loads(claude.content[0].text) if claude.content else {})
            ),
            output=Rating,
        )
        chain.save("ratings")
        print(chain)
        ```
    """

    DEFAULT_FILE_RECORD: ClassVar[dict] = {
        "id": 0,
        "source": "",
        "name": "",
        "vtype": "",
        "size": 0,
        "random": 0,
    }

    def __init__(self, *args, **kwargs):
        """This method needs to be redefined as a part of Dataset and DacaChin
        decoupling."""
        super().__init__(
            *args,
            **kwargs,
            indexing_column_types=File._datachain_column_types,
        )
        self._settings = Settings()
        self._setup = {}

        if self.feature_schema:
            self.signals_schema = SignalSchema.deserialize(self.feature_schema)
        else:
            self.signals_schema = SignalSchema.from_column_types(self.column_types)

    @property
    def schema(self):
        return self.signals_schema.values if self.signals_schema else None

    def print_schema(self):
        self.signals_schema.print_tree()

    def settings(
        self, cache=None, batch=None, parallel=None, workers=None, min_task_size=None
    ) -> "Self":
        """Change settings for chain.

        This function changes specified settings without changing not specified ones.
        It returns chain, so, it can be chained later with next operation.

        Parameters:
            cache : data caching (default=False)
            batch : size of the batch (default=1000)
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
        self._settings.add(Settings(cache, batch, parallel, workers, min_task_size))
        return self

    def reset_settings(self, settings: Optional[Settings] = None) -> "Self":
        """Reset all settings to default values."""
        self._settings = settings if settings else Settings()
        return self

    def reset_schema(self, signals_schema: SignalSchema) -> "Self":
        self.signals_schema = signals_schema
        return self

    def add_schema(self, signals_schema: SignalSchema) -> "Self":
        union = self.signals_schema.values | signals_schema.values
        self.signals_schema = SignalSchema(union)
        return self

    def get_file_signals(self) -> list[str]:
        return self.signals_schema.get_file_signals()

    @classmethod
    def from_storage(
        cls,
        path,
        *,
        type: Literal["binary", "text", "image"] = "binary",
        catalog: Optional["Catalog"] = None,
        recursive: Optional[bool] = True,
        anon: bool = False,
    ) -> "Self":
        """Get data from a storage as a list of file with all file attributes. It
        returns the chain itself as usual.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            recursive : search recursively for the given path.
            anon : use anonymous mode to access the storage.

        Example:
            ```py
            chain = DataChain.from_storage("s3://my-bucket/my-dir")
            ```
        """
        func = get_file(type)
        return cls(path, catalog=catalog, recursive=recursive, anon=anon).map(file=func)

    @classmethod
    def from_dataset(cls, name: str, version: Optional[int] = None) -> "DataChain":
        """Get data from dataset. It returns the chain itself.

        Parameters:
            name : dataset name
            version : dataset version

        Examples:
            >>> chain = DataChain.from_dataset("my_cats")
        """
        return DataChain(name=name, version=version)

    @classmethod
    def from_csv(
        cls,
        path,
        type: Literal["binary", "text", "image"] = "text",
        anon: bool = False,
        spec: Optional[FeatureType] = None,
        schema_from: Optional[str] = "auto",
        object_name: Optional[str] = "csv",
        model_name: Optional[str] = None,
        show_schema: Optional[bool] = False,
    ) -> "DataChain":
        """Get data from CSV. It returns the chain itself.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "text".
            anon : use anonymous mode to access the storage.
            spec : Data Model for CSV file
            object_name : generated object column name
            model_name : generated model name
            schema_from : path to sample to infer spec from
            show_schema : print auto-generated schema

        Examples:
            infer model from the first two lines (header + data)
            >>> chain = DataChain.from_csv("gs://csv")

            use a particular data model
            >>> chain = DataChain.from_csv("gs://csv"i, spec=MyModel)
        """
        if schema_from == "auto":
            schema_from = path

        chain = DataChain.from_storage(path=path, type=type, anon=anon)
        signal_dict = {
            object_name: read_meta(
                schema_from=schema_from,
                meta_type="csv",
                spec=spec,
                model_name=model_name,
                show_schema=show_schema,
            )
        }
        return chain.gen(**signal_dict)  # type: ignore[misc, arg-type]

    @classmethod
    def from_json(
        cls,
        path,
        type: Literal["binary", "text", "image"] = "text",
        anon: bool = False,
        spec: Optional[FeatureType] = None,
        schema_from: Optional[str] = "auto",
        jmespath: Optional[str] = None,
        object_name: Optional[str] = None,
        model_name: Optional[str] = None,
        show_schema: Optional[bool] = False,
        meta_type: Optional[str] = "json",
    ) -> "DataChain":
        """Get data from JSON. It returns the chain itself.

        Parameters:
            path : storage URI with directory. URI must start with storage prefix such
                as `s3://`, `gs://`, `az://` or "file:///"
            type : read file as "binary", "text", or "image" data. Default is "binary".
            anon : use anonymous mode to access the storage.
            spec : optional Data Model
            schema_from : path to sample to infer spec from
            object_name : generated object column name
            model_name : generated model name
            show_schema : print auto-generated schema
            jmespath : JMESPATH expression to reduce JSON

        Examples:
            infer JSON schema from data, reduce using JMESPATH, print schema
            >>> chain = DataChain.from_json("gs://json", jmespath="key1.key2")

            infer JSON schema from a particular path, print data model
            >>> chain = DataChain.from_json("gs://json_ds", schema_from="gs://json/my.json")
        """
        if schema_from == "auto":
            schema_from = path

        def jmespath_to_name(s: str):
            name_end = re.search(r"\W", s).start() if re.search(r"\W", s) else len(s)  # type: ignore[union-attr]
            return s[:name_end]

        if (not object_name) and jmespath:
            object_name = jmespath_to_name(jmespath)
        if not object_name:
            object_name = "json"
        chain = DataChain.from_storage(path=path, type=type, anon=anon)
        signal_dict = {
            object_name: read_meta(
                schema_from=schema_from,
                meta_type=meta_type,
                spec=spec,
                model_name=model_name,
                show_schema=show_schema,
                jmespath=jmespath,
            )
        }
        return chain.gen(**signal_dict)  # type: ignore[arg-type]

    def show_json_schema(  # type: ignore[override]
        self, jmespath: Optional[str] = None, model_name: Optional[str] = None
    ) -> "DataChain":
        """Print JSON data model and save it. It returns the chain itself.

        Parameters:
            jmespath : JMESPATH expression to reduce JSON
            model_name : generated model name

        Examples:
            print JSON schema and save to column "meta_from":
            >>> uri = "gs://datachain-demo/coco2017/annotations_captions/"
            >>> chain = DataChain.from_storage(uri)
            >>> chain = chain.show_json_schema()
            >>> chain.save()
        """
        return self.map(
            meta_schema=lambda file: read_schema(
                file, data_type="json", expr=jmespath, model_name=model_name
            ),
            output=str,
        )

    def show_jsonl_schema(  # type: ignore[override]
        self, jmespath: Optional[str] = None, model_name: Optional[str] = None
    ) -> "DataChain":
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
    ) -> "DataChain":
        """Save to a Dataset. It returns the chain itself.

        Parameters:
            name : dataset name. Empty name saves to a temporary dataset that will be
                removed after process ends. Temp dataset are useful for optimization.
            version : version of a dataset. Default - the last version that exist.
        """
        schema = self.signals_schema.serialize()
        return super().save(name=name, version=version, feature_schema=schema)

    def apply(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def map(
        self,
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
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

        Examples:
            Using signal_map and single type in output:
            >>> chain = chain.map(value=lambda name: name[:-4] + ".json", output=str)
            >>> chain.save("new_dataset")

            Using func and output as a map:
            >>> chain = chain.map(lambda name: name[:-4] + ".json", output={"res": str})
            >>> chain.save("new_dataset")
        """

        udf_obj = self._udf_to_obj(Mapper, func, params, output, signal_map)

        chain = self.add_signals(
            udf_obj.to_udf_wrapper(self._settings.batch),
            **self._settings.to_dict(),
        )

        return chain.add_schema(udf_obj.output).reset_settings(self._settings)

    def gen(
        self,
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
        **signal_map,
    ) -> "Self":
        """Apply a function to each row to create new rows (with potentially new
        signals). The function needs to return a new objects for each of the new rows.
        It returns a chain itself with new signals.

        Input-output relationship: 1:N

        This method is similar to `map()`, uses the same list of parameters, but with
        one key differences: It produces a sequence of rows for each input row (like
        extracting multiple file records from a single tar file or bounding boxes from a
        single image file).
        """

        udf_obj = self._udf_to_obj(Generator, func, params, output, signal_map)
        chain = DatasetQuery.generate(
            self,
            udf_obj.to_udf_wrapper(self._settings.batch),
            **self._settings.to_dict(),
        )

        return chain.reset_schema(udf_obj.output).reset_settings(self._settings)

    def agg(
        self,
        func: Optional[Callable] = None,
        partition_by: Optional[PartitionByType] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
        **signal_map,
    ) -> "Self":
        """Aggregate rows using `partition_by` statement and apply a function to the
        groups of aggregated rows. The function needs to return new objects for each
        group of the new rows. It returns a chain itself with new signals.

        Input-output relationship: N:M

        This method bears similarity to `gen()` and map(), employing a comparable set of
        parameters, yet differs in two crucial aspects:
        1. The `partition_by` parameter: This specifies the column name or a list of
           column names that determine the grouping criteria for aggregation.
        2. Group-based UDF function input: Instead of individual rows, the function
           receives a list all rows within each group defined by `partition_by`.
        """
        udf_obj = self._udf_to_obj(Aggregator, func, params, output, signal_map)
        chain = DatasetQuery.generate(
            self,
            udf_obj.to_udf_wrapper(self._settings.batch),
            partition_by=partition_by,
            **self._settings.to_dict(),
        )

        return chain.reset_schema(udf_obj.output).reset_settings(self._settings)

    def batch_map(
        self,
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
        **signal_map,
    ) -> "Self":
        """This is a batch version of map().

        It accepts the same parameters plus an
        additional parameter:
        """
        udf_obj = self._udf_to_obj(BatchMapper, func, params, output, signal_map)
        chain = DatasetQuery.generate(
            self,
            udf_obj.to_udf_wrapper(self._settings.batch),
            **self._settings.to_dict(),
        )

        return chain.add_schema(udf_obj.output).reset_settings(self._settings)

    def _udf_to_obj(
        self,
        target_class: type[UDFBase],
        func: Optional[Callable],
        params: Union[None, str, Sequence[str]],
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]],
        signal_map,
    ) -> UDFBase:
        is_generator = target_class.is_output_batched
        name = self.name or ""

        sign = UdfSignature.parse(name, signal_map, func, params, output, is_generator)
        params_schema = self.signals_schema.slice(sign.params, self._setup)

        return UDFBase._create(target_class, sign, params_schema)

    def _extend_features(self, method_name, *args, **kwargs):
        super_func = getattr(super(), method_name)

        new_schema = self.signals_schema.resolve(*args)
        columns = [C(col) for col in new_schema.db_signals()]
        res = super_func(*columns, **kwargs)
        if isinstance(res, DataChain):
            res.signals_schema = new_schema

        return res

    @detach
    def select(self, *args: str) -> "Self":
        """Select only a specified set of signals."""
        new_schema = self.signals_schema.resolve(*args)
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

    def iterate(self, *cols: str) -> Iterator[list[FeatureType]]:
        """Iterate over rows.

        If columns are specified - limit them to specified
        columns.
        """
        chain = self.select(*cols) if cols else self

        db_signals = chain.signals_schema.db_signals()
        with super().select(*db_signals).as_iterable() as rows_iter:
            for row in rows_iter:
                yield chain.signals_schema.row_to_features(row, chain.session.catalog)

    def iterate_one(self, col: str) -> Iterator[FeatureType]:
        for item in self.iterate(col):
            yield item[0]

    def collect(self, *cols: str) -> list[list[FeatureType]]:
        return list(self.iterate(*cols))

    def collect_one(self, col: str) -> list[FeatureType]:
        return list(self.iterate_one(col))

    def to_pytorch(self, **kwargs):
        """Convert to pytorch dataset format."""

        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Missing required dependency 'torch' for Dataset.to_pytorch()"
            ) from exc
        from datachain.lib.pytorch import PytorchDataset

        if self.attached:
            chain = self
        else:
            chain = self.save()
        assert chain.name is not None  # for mypy
        return PytorchDataset(chain.name, chain.version, catalog=self.catalog, **kwargs)

    def remove_file_signals(self) -> "Self":
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

        Examples:
            >>> meta = meta_emd.merge(meta_pq, on=(C.name, C.emd__index),
                                    right_on=(C.name, C.pq__index))
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

        on_columns = self.signals_schema.resolve(*on).db_signals()

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

            right_on_columns = right_ds.signals_schema.resolve(*right_on).db_signals()

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
        ds.signals_schema = self.signals_schema.merge(right_ds.signals_schema, rname)

        return ds

    @classmethod
    def from_features(
        cls,
        ds_name: str = "",
        session: Optional[Session] = None,
        output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
        **fr_map,
    ) -> "DataChain":
        """Generate chain from list of features."""
        tuple_type, output, tuples = features_to_tuples(ds_name, output, **fr_map)

        def _func_fr() -> Iterator[tuple_type]:  # type: ignore[valid-type]
            yield from tuples

        chain = DataChain.create_empty(DataChain.DEFAULT_FILE_RECORD, session=session)
        return chain.gen(_func_fr, output=output)

    @classmethod
    def from_pandas(  # type: ignore[override]
        cls, df: "pd.DataFrame", name: str = "", session: Optional[Session] = None
    ) -> "DataChain":
        """Generate chain from pandas data-frame."""
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

        return cls.from_features(name, session, **fr_map)

    def parse_tabular(
        self,
        output: Optional[dict[str, FeatureType]] = None,
        **kwargs,
    ) -> "DataChain":
        """Generate chain from list of tabular files.

        Parameters:
            output : Dictionary defining column names and their corresponding types.
            kwargs : Parameters to pass to pyarrow.dataset.dataset.

        Examples:
            Reading a json lines file:
            >>> dc = DataChain.from_storage("s3://mybucket/file.jsonl")
            >>> dc = dc.parse_tabular(format="json")

            Reading a filtered list of files as a dataset:
            >>> dc = DataChain.from_storage("s3://mybucket")
            >>> dc = dc.filter(C("file.name").glob("*.jsonl"))
            >>> dc = dc.parse_tabular(format="json")
        """
        from pyarrow import unify_schemas
        from pyarrow.dataset import dataset

        from datachain.lib.arrow import ArrowGenerator, schema_to_output

        schema = None
        if output:
            output = {"source": IndexedFile} | output
        else:
            schemas = []
            for row in self.select("file").iterate():
                file = row[0]
                ds = dataset(file.get_path(), filesystem=file.get_fs(), **kwargs)  # type: ignore[union-attr]
                schemas.append(ds.schema)
            if not schemas:
                msg = "error parsing tabular data schema - found no files to parse"
                raise DatasetPrepareError(self.name, msg)
            schema = unify_schemas(schemas)
            try:
                output = schema_to_output(schema)
            except ValueError as e:
                raise DatasetPrepareError(self.name, e) from e

        return self.gen(ArrowGenerator(schema, **kwargs), output=output)

    def parse_csv(
        self,
        delimiter: str = ",",
        header: bool = True,
        column_names: Optional[list[str]] = None,
        output: Optional[dict[str, FeatureType]] = None,
    ) -> "DataChain":
        """Generate chain from list of csv files.

        Parameters:
            delimiter : Character for delimiting columns.
            header : Whether the files include a header row.
            column_names : Column names if no header. Implies `header = False`.
            output : Dictionary defining column names and their corresponding types.

        Examples:
            Reading a csv file:
            >>> dc = DataChain.from_storage("s3://mybucket/file.csv")
            >>> dc = dc.parse_tabular(format="csv")

            Reading a filtered list of csv files as a dataset:
            >>> dc = DataChain.from_storage("s3://mybucket")
            >>> dc = dc.filter(C("file.name").glob("*.csv"))
            >>> dc = dc.parse_tabular()
        """
        from pyarrow.csv import ParseOptions, ReadOptions
        from pyarrow.dataset import CsvFileFormat

        if column_names and output:
            msg = "error parsing csv - only one of column_names or output is allowed"
            raise DatasetPrepareError(self.name, msg)

        if not header and not column_names:
            if output:
                column_names = list(output.keys())
            else:
                msg = "error parsing csv - provide column_names or output if no header"
                raise DatasetPrepareError(self.name, msg)

        parse_options = ParseOptions(delimiter=delimiter)
        read_options = ReadOptions(column_names=column_names)
        format = CsvFileFormat(parse_options=parse_options, read_options=read_options)
        return self.parse_tabular(output=output, format=format)

    def parse_parquet(
        self,
        partitioning: Any = "hive",
        output: Optional[dict[str, FeatureType]] = None,
    ) -> "DataChain":
        """Generate chain from list of parquet files.

        Parameters:
            partitioning : Any pyarrow partitioning schema.
            output : Dictionary defining column names and their corresponding types.

        Examples:
            Reading a single file:
            >>> dc = DataChain.from_storage("s3://mybucket/file.parquet")
            >>> dc = dc.parse_tabular()

            Reading a partitioned dataset from a directory:
            >>> dc = DataChain.from_storage("path/to/dir")
            >>> dc = dc.parse_tabular()

            Reading a filtered list of files as a dataset:
            >>> dc = DataChain.from_storage("s3://mybucket")
            >>> dc = dc.filter(C("file.name").glob("*.parquet"))
            >>> dc = dc.parse_tabular()

            Reading a filtered list of partitions as a dataset:
            >>> dc = DataChain.from_storage("s3://mybucket")
            >>> dc = dc.filter(C("file.parent").glob("*month=1*"))
            >>> dc = dc.parse_tabular()
        """
        return self.parse_tabular(
            output=output, format="parquet", partitioning=partitioning
        )

    @classmethod
    def create_empty(
        cls,
        to_insert: Optional[Union[dict, list[dict]]],
        session: Optional[Session] = None,
    ) -> "DataChain":
        """Create empty chain. Returns a chain. This method is used for programmatically
        generating a chains in contrast of reading data from storages or other sources.

        Parameters:
            to_insert : records (or a single record) to insert. Each record is
                        a dictionary of signals and theirs values.

        Examples:
            >>> empty = DataChain.create_empty()
            >>> single_record = DataChain.create_empty(DataChain.DEFAULT_FILE_RECORD)
        """
        session = Session.get(session)
        catalog = session.catalog

        name = session.generate_temp_dataset_name()
        columns: tuple[sqlalchemy.Column[Any], ...] = tuple(
            sqlalchemy.Column(name, typ)
            for name, typ in File._datachain_column_types.items()
        )
        dsr = catalog.create_dataset(name, columns=columns)

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
        return DataChain(name=dsr.name)

    def sum(self, fr: FeatureType):  # type: ignore[override]
        return self._extend_features("sum", fr)

    def avg(self, fr: FeatureType):  # type: ignore[override]
        return self._extend_features("avg", fr)

    def min(self, fr: FeatureType):  # type: ignore[override]
        return self._extend_features("min", fr)

    def max(self, fr: FeatureType):  # type: ignore[override]
        return self._extend_features("max", fr)

    def setup(self, **kwargs) -> "Self":
        intersection = set(self._setup.keys()) & set(kwargs.keys())
        if intersection:
            keys = ", ".join(intersection)
            raise DatasetPrepareError(self.name, f"this value(s) already setup: {keys}")

        self._setup = self._setup | kwargs
        return self
