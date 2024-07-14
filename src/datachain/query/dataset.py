import ast
import contextlib
import datetime
import inspect
import json
import logging
import os
import random
import re
import string
import subprocess
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator, Sequence
from copy import copy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import attrs
import pandas as pd
import sqlalchemy
from attrs import frozen
from dill import dumps, source
from fsspec.callbacks import DEFAULT_CALLBACK, Callback, TqdmCallback
from pydantic import BaseModel
from sqlalchemy import Column
from sqlalchemy.sql import func as f
from sqlalchemy.sql.elements import ColumnClause, ColumnElement
from sqlalchemy.sql.expression import label
from sqlalchemy.sql.schema import TableClause
from sqlalchemy.sql.selectable import Select

from datachain.asyn import ASYNC_WORKERS, AsyncMapper, OrderedMapper
from datachain.catalog import (
    QUERY_SCRIPT_CANCELED_EXIT_CODE,
    QUERY_SCRIPT_INVALID_LAST_STATEMENT_EXIT_CODE,
    get_catalog,
)
from datachain.data_storage.schema import (
    PARTITION_COLUMN_ID,
    partition_col_names,
    partition_columns,
)
from datachain.dataset import DatasetStatus, RowDict
from datachain.error import DatasetNotFoundError, QueryScriptCancelError
from datachain.progress import CombinedDownloadCallback
from datachain.query.schema import DEFAULT_DELIMITER
from datachain.sql.functions import rand
from datachain.storage import Storage, StorageURI
from datachain.utils import batched, determine_processes, inside_notebook

from .batch import RowBatch
from .metrics import metrics
from .schema import C, UDFParamSpec, normalize_param
from .session import Session
from .udf import UDFBase, UDFClassWrapper, UDFFactory, UDFType

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ClauseElement
    from sqlalchemy.sql.schema import Table
    from sqlalchemy.sql.selectable import GenerativeSelect
    from typing_extensions import Concatenate, ParamSpec, Self

    from datachain.catalog import Catalog
    from datachain.data_storage import AbstractWarehouse
    from datachain.dataset import DatasetRecord

    from .udf import UDFResult

    P = ParamSpec("P")


INSERT_BATCH_SIZE = 10000

PartitionByType = Union[ColumnElement, Sequence[ColumnElement]]
JoinPredicateType = Union[str, ColumnClause, ColumnElement]
# dependency can be either dataset_name + dataset_version tuple or just storage uri
# depending what type of dependency we are adding
DatasetDependencyType = Union[tuple[str, int], StorageURI]

logger = logging.getLogger("datachain")


T = TypeVar("T", bound="DatasetQuery")


def detach(
    method: "Callable[Concatenate[T, P], T]",
) -> "Callable[Concatenate[T, P], T]":
    """
    Decorator that needs to be put on a method that modifies existing DatasetQuery
    which was 100% representing one particular dataset and had name and version of
    that dataset set, and which returns new instance of it.
    This kind of DatasetQuery, which represent one whole dataset, we return from
    .save() method.
    Example of modifying method is .filter() as that one filters out part
    of a dataset which means DatasetQuery no longer 100% represents it (in this case
    it can represents only a part of it)
    """

    @wraps(method)
    def _inner(self: T, *args: "P.args", **kwargs: "P.kwargs") -> T:
        cloned = method(self, *args, **kwargs)
        cloned.name = None
        cloned.version = None
        return cloned

    return _inner


class QueryGeneratorFunc(Protocol):
    def __call__(self, *columns: ColumnElement) -> Select: ...


@frozen
class QueryGenerator:
    func: QueryGeneratorFunc
    columns: tuple[ColumnElement, ...]

    def exclude(self, column_names) -> Select:
        return self.func(*(c for c in self.columns if c.name not in column_names))

    def select(self, column_names=None) -> Select:
        if column_names is None:
            return self.func(*self.columns)
        return self.func(*(c for c in self.columns if c.name in column_names))


@frozen
class StepResult:
    query_generator: QueryGenerator
    dependencies: tuple[DatasetDependencyType, ...]


def step_result(
    func: QueryGeneratorFunc,
    columns: Iterable[ColumnElement],
    dependencies: Iterable[DatasetDependencyType] = (),
) -> "StepResult":
    return StepResult(
        query_generator=QueryGenerator(func=func, columns=tuple(columns)),
        dependencies=tuple(dependencies),
    )


class StartingStep(ABC):
    """An initial query processing step, referencing a data source."""

    @abstractmethod
    def apply(self) -> "StepResult": ...


@frozen
class Step(ABC):
    """A query processing step (filtering, mutation, etc.)"""

    @abstractmethod
    def apply(
        self, query_generator: "QueryGenerator", temp_tables: list[str]
    ) -> "StepResult":
        """Apply the processing step."""


@frozen
class QueryStep(StartingStep):
    catalog: "Catalog"
    dataset_name: str
    dataset_version: int

    def apply(self):
        def q(*columns):
            return sqlalchemy.select(*columns)

        dataset = self.catalog.get_dataset(self.dataset_name)
        table = self.catalog.warehouse.dataset_rows(dataset, self.dataset_version)

        return step_result(
            q, table.c, dependencies=[(self.dataset_name, self.dataset_version)]
        )


@frozen
class IndexingStep(StartingStep):
    path: str
    catalog: "Catalog"
    kwargs: dict[str, Any]
    recursive: Optional[bool] = True

    def apply(self):
        self.catalog.index([self.path], **self.kwargs)
        uri, path = self.parse_path()
        _partial_id, partial_path = self.catalog.metastore.get_valid_partial_id(
            uri, path
        )
        dataset = self.catalog.get_dataset(Storage.dataset_name(uri, partial_path))
        dataset_rows = self.catalog.warehouse.dataset_rows(
            dataset, dataset.latest_version
        )

        def q(*columns):
            col_names = [c.name for c in columns]
            return self.catalog.warehouse.nodes_dataset_query(
                dataset_rows,
                column_names=col_names,
                path=path,
                recursive=self.recursive,
            )

        storage = self.catalog.get_storage(uri)

        return step_result(q, dataset_rows.c, dependencies=[storage.uri])

    def parse_path(self):
        client_config = self.kwargs.get("client_config") or {}
        client, path = self.catalog.parse_url(self.path, **client_config)
        return client.uri, path


def generator_then_call(generator, func: Callable):
    """
    Yield items from generator then execute a function and yield
    its result.
    """
    yield from generator
    yield func() or []


@frozen
class DatasetDiffOperation(Step):
    """
    Abstract class for operations that are calculation some kind of diff between
    datasets queries like subtract, changed etc.
    """

    dq: "DatasetQuery"
    catalog: "Catalog"

    def clone(self) -> "Self":
        return self.__class__(self.dq, self.catalog)

    @abstractmethod
    def query(
        self,
        source_query: Select,
        target_query: Select,
    ) -> Select:
        """
        Should return select query that calculates desired diff between dataset queries
        """

    def apply(self, query_generator, temp_tables: list[str]):
        source_query = query_generator.exclude(("id",))
        target_query = self.dq.apply_steps().select()
        temp_tables.extend(self.dq.temp_table_names)

        # creating temp table that will hold subtract results
        temp_table_name = self.catalog.warehouse.TMP_TABLE_NAME_PREFIX + _random_string(
            6
        )
        temp_tables.append(temp_table_name)

        columns = [
            c if isinstance(c, Column) else Column(c.name, c.type)
            for c in source_query.columns
        ]
        temp_table = self.catalog.warehouse.create_dataset_rows_table(
            temp_table_name,
            columns=columns,
            if_not_exists=False,
        )

        diff_q = self.query(source_query, target_query)

        insert_q = temp_table.insert().from_select(
            source_query.selected_columns, diff_q
        )

        self.catalog.warehouse.db.execute(insert_q)

        def q(*columns):
            return sqlalchemy.select(*columns)

        return step_result(q, temp_table.c)


@frozen
class Subtract(DatasetDiffOperation):
    """
    Calculates rows that are in a source query but are not in target query (diff)
    This can be used to do delta updates (calculate UDF only on newly added rows)
    Example:
        >>> ds = DatasetQuery(name="dogs_cats") # some older dataset with embeddings
        >>> ds_updated = (
                DatasetQuery("gs://dvcx-datalakes/dogs-and-cats")
                .filter(C.size > 1000) # we can also filter out source query
                .subtract(ds)
                .add_signals(calc_embeddings) # calculae embeddings only on new rows
                .union(ds) # union with old dataset that's missing new rows
                .save("dogs_cats_updated")
            )
    """

    def query(self, source_query: Select, target_query: Select) -> Select:
        return self.catalog.warehouse.subtract_query(source_query, target_query)


@frozen
class Changed(DatasetDiffOperation):
    """
    Calculates rows that are changed in a source query compared to target query
    Changed means it has same source + parent + name but different last_modified
    Example:
        >>> ds = DatasetQuery(name="dogs_cats") # some older dataset with embeddings
        >>> ds_updated = (
                DatasetQuery("gs://dvcx-datalakes/dogs-and-cats")
                .filter(C.size > 1000) # we can also filter out source query
                .changed(ds)
                .add_signals(calc_embeddings) # calculae embeddings only on changed rows
                .union(ds) # union with old dataset that's missing updated rows
                .save("dogs_cats_updated")
            )

    """

    def query(self, source_query: Select, target_query: Select) -> Select:
        return self.catalog.warehouse.changed_query(source_query, target_query)


def adjust_outputs(
    warehouse: "AbstractWarehouse", row: dict[str, Any], udf_col_types: list[tuple]
) -> dict[str, Any]:
    """
    This function does a couple of things to prepare a row for inserting into the db:
    1. Fill default values for columns that have None and add missing columns
    2. Validate values with its corresponding DB column types and convert types
       if needed and possible
    """
    # Optimization: Use precomputed column type values as these do not change for each
    # row in the same UDF.
    for (
        col_name,
        col_type,
        col_python_type,
        col_type_name,
        default_value,
    ) in udf_col_types:
        row_val = row.get(col_name)

        # Fill None or missing values with defaults (get returns None if not in the row)
        if row_val is None:
            row[col_name] = default_value
            continue

        # Validate and convert type if needed and possible
        row[col_name] = warehouse.convert_type(
            row_val, col_type, col_python_type, col_type_name, col_name
        )
    return row


def get_udf_col_types(warehouse: "AbstractWarehouse", udf: UDFBase) -> list[tuple]:
    """Optimization: Precompute UDF column types so these don't have to be computed
    in the convert_type function for each row in a loop."""
    dialect = warehouse.db.dialect
    return [
        (
            col_name,
            # Check if type is already instantiated or not
            col_type_inst := col_type() if inspect.isclass(col_type) else col_type,
            warehouse.python_type(col_type_inst),
            type(col_type_inst).__name__,
            col_type.default_value(dialect),
        )
        for col_name, col_type in udf.output.items()
    ]


def process_udf_outputs(
    warehouse: "AbstractWarehouse",
    udf_table: "Table",
    udf_results: Iterator[Iterable["UDFResult"]],
    udf: UDFBase,
    batch_size=INSERT_BATCH_SIZE,
    cb: Callback = DEFAULT_CALLBACK,
) -> None:
    rows: list[UDFResult] = []
    # Optimization: Compute row types once, rather than for every row.
    udf_col_types = get_udf_col_types(warehouse, udf)

    for udf_output in udf_results:
        if not udf_output:
            continue
        for row in udf_output:
            cb.relative_update()
            rows.append(adjust_outputs(warehouse, row, udf_col_types))
            if len(rows) >= batch_size:
                for row_chunk in batched(rows, batch_size):
                    warehouse.insert_rows(udf_table, row_chunk)
                rows.clear()

    if rows:
        for row_chunk in batched(rows, batch_size):
            warehouse.insert_rows(udf_table, row_chunk)


def get_download_callback() -> Callback:
    return CombinedDownloadCallback(
        {"desc": "Download", "unit": "B", "unit_scale": True, "unit_divisor": 1024}
    )


def get_processed_callback() -> Callback:
    return TqdmCallback({"desc": "Processed", "unit": " rows"})


def get_generated_callback(is_generator: bool = False) -> Callback:
    if is_generator:
        return TqdmCallback({"desc": "Generated", "unit": " rows"})
    return DEFAULT_CALLBACK


def run_udf(
    udf,
    udf_inputs,
    catalog,
    is_generator,
    cache,
    download_cb: Callback = DEFAULT_CALLBACK,
    processed_cb: Callback = DEFAULT_CALLBACK,
) -> Iterator[Iterable["UDFResult"]]:
    for batch in udf_inputs:
        n_rows = len(batch.rows) if isinstance(batch, RowBatch) else 1
        output = udf(catalog, batch, is_generator, cache, cb=download_cb)
        processed_cb.relative_update(n_rows)
        yield output


@frozen
class UDF(Step, ABC):
    udf: UDFType
    catalog: "Catalog"
    partition_by: Optional[PartitionByType] = None
    parallel: Optional[int] = None
    workers: Union[bool, int] = False
    min_task_size: Optional[int] = None
    is_generator = False
    cache: bool = False

    @abstractmethod
    def create_udf_table(self, query: Select) -> "Table":
        """Method that creates a table where temp udf results will be saved"""

    def process_input_query(self, query: Select) -> tuple[Select, list["Table"]]:
        """Apply any necessary processing to the input query"""
        return query, []

    @abstractmethod
    def create_result_query(
        self, udf_table: "Table", query: Select
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        """
        Method that should return query to fetch results from udf and columns
        to select
        """

    def udf_table_name(self) -> str:
        return self.catalog.warehouse.UDF_TABLE_NAME_PREFIX + _random_string(6)

    def populate_udf_table(self, udf_table: "Table", query: Select) -> None:
        use_partitioning = self.partition_by is not None
        batching = self.udf.properties.get_batching(use_partitioning)
        workers = self.workers
        if (
            not workers
            and os.environ.get("DATACHAIN_DISTRIBUTED")
            and os.environ.get("DATACHAIN_SETTINGS_WORKERS")
        ):
            # Enable distributed processing by default if the module is available,
            # and a default number of workers is provided.
            workers = True

        processes = determine_processes(self.parallel)

        try:
            if workers:
                from datachain.catalog.loader import get_distributed_class

                distributor = get_distributed_class(min_task_size=self.min_task_size)
                distributor(
                    self.udf,
                    self.catalog,
                    udf_table,
                    query,
                    workers,
                    processes,
                    is_generator=self.is_generator,
                    use_partitioning=use_partitioning,
                    cache=self.cache,
                )
            elif processes:
                # Parallel processing (faster for more CPU-heavy UDFs)
                udf_info = {
                    "udf": self.udf,
                    "catalog_init": self.catalog.get_init_params(),
                    "id_generator_clone_params": (
                        self.catalog.id_generator.clone_params()
                    ),
                    "metastore_clone_params": self.catalog.metastore.clone_params(),
                    "warehouse_clone_params": self.catalog.warehouse.clone_params(),
                    "table": udf_table,
                    "query": query,
                    "batching": batching,
                    "processes": processes,
                    "is_generator": self.is_generator,
                    "cache": self.cache,
                }

                # Run the UDFDispatcher in another process to avoid needing
                # if __name__ == '__main__': in user scripts
                datachain_exec_path = os.environ.get("DATACHAIN_EXEC_PATH", "datachain")

                envs = dict(os.environ)
                envs.update({"PYTHONPATH": os.getcwd()})
                with self.process_feature_module():
                    process_data = dumps(udf_info, recurse=True)
                    result = subprocess.run(  # noqa: S603
                        [datachain_exec_path, "--internal-run-udf"],
                        input=process_data,
                        check=False,
                        env=envs,
                    )
                    if result.returncode != 0:
                        raise RuntimeError("UDF Execution Failed!")

            else:
                # Otherwise process single-threaded (faster for smaller UDFs)
                # Optionally instantiate the UDF instance if a class is provided.
                if isinstance(self.udf, UDFFactory):
                    udf: UDFBase = self.udf()
                else:
                    udf = self.udf

                if hasattr(udf.func, "setup") and callable(udf.func.setup):
                    udf.func.setup()

                warehouse = self.catalog.warehouse

                with contextlib.closing(
                    batching(warehouse.dataset_select_paginated, query)
                ) as udf_inputs:
                    download_cb = get_download_callback()
                    processed_cb = get_processed_callback()
                    generated_cb = get_generated_callback(self.is_generator)
                    try:
                        udf_results = run_udf(
                            udf,
                            udf_inputs,
                            self.catalog,
                            self.is_generator,
                            self.cache,
                            download_cb,
                            processed_cb,
                        )
                        process_udf_outputs(
                            warehouse,
                            udf_table,
                            udf_results,
                            udf,
                            cb=generated_cb,
                        )
                    finally:
                        download_cb.close()
                        processed_cb.close()
                        generated_cb.close()

                warehouse.insert_rows_done(udf_table)

                if hasattr(udf.func, "teardown") and callable(udf.func.teardown):
                    udf.func.teardown()

        except QueryScriptCancelError:
            self.catalog.warehouse.close()
            sys.exit(QUERY_SCRIPT_CANCELED_EXIT_CODE)
        except (Exception, KeyboardInterrupt):
            # Close any open database connections if an error is encountered
            self.catalog.warehouse.close()
            raise

    @contextlib.contextmanager
    def process_feature_module(self):
        # Generate a random name for the feature module
        feature_module_name = "tmp" + _random_string(10)
        # Create a dynamic module with the generated name
        dynamic_module = types.ModuleType(feature_module_name)
        # Get the import lines for the necessary objects from the main module
        main_module = sys.modules["__main__"]
        if getattr(main_module, "__file__", None):
            import_lines = list(get_imports(main_module))
        else:
            import_lines = [
                source.getimport(obj, alias=name)
                for name, obj in main_module.__dict__.items()
                if _imports(obj) and not (name.startswith("__") and name.endswith("__"))
            ]

        # Get the feature classes from the main module
        feature_classes = {
            name: obj
            for name, obj in main_module.__dict__.items()
            if _feature_predicate(obj)
        }
        if not feature_classes:
            yield None
            return

        # Get the source code of the feature classes
        feature_sources = [source.getsource(cls) for _, cls in feature_classes.items()]
        # Set the module name for the feature classes to the generated name
        for name, cls in feature_classes.items():
            cls.__module__ = feature_module_name
            setattr(dynamic_module, name, cls)
        # Add the dynamic module to the sys.modules dictionary
        sys.modules[feature_module_name] = dynamic_module
        # Combine the import lines and feature sources
        feature_file = "\n".join(import_lines) + "\n" + "\n".join(feature_sources)

        # Write the module content to a .py file
        with open(f"{feature_module_name}.py", "w") as module_file:
            module_file.write(feature_file)

        try:
            yield feature_module_name
        finally:
            for cls in feature_classes.values():
                cls.__module__ = main_module.__name__
            os.unlink(f"{feature_module_name}.py")
            # Remove the dynamic module from sys.modules
            del sys.modules[feature_module_name]

    def create_partitions_table(self, query: Select) -> "Table":
        """
        Create temporary table with group by partitions.
        """
        assert self.partition_by is not None

        if isinstance(self.partition_by, Sequence):
            list_partition_by = self.partition_by
        else:
            list_partition_by = [self.partition_by]

        # create table with partitions
        tbl = self.catalog.warehouse.create_udf_table(
            self.udf_table_name(), partition_columns()
        )

        # fill table with partitions
        cols = [
            query.selected_columns.id,
            f.dense_rank().over(order_by=list_partition_by).label(PARTITION_COLUMN_ID),
        ]
        self.catalog.warehouse.db.execute(
            tbl.insert().from_select(cols, query.with_only_columns(*cols))
        )

        return tbl

    def clone(self, partition_by: Optional[PartitionByType] = None) -> "Self":
        if partition_by is not None:
            return self.__class__(
                self.udf,
                self.catalog,
                partition_by=partition_by,
                parallel=self.parallel,
                workers=self.workers,
                min_task_size=self.min_task_size,
            )
        return self.__class__(self.udf, self.catalog)

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> "StepResult":
        _query = query = query_generator.select()

        # Apply partitioning if needed.
        if self.partition_by is not None:
            partition_tbl = self.create_partitions_table(query)
            temp_tables.append(partition_tbl.name)

            subq = query.subquery()
            query = (
                sqlalchemy.select(*subq.c)
                .outerjoin(partition_tbl, partition_tbl.c.id == subq.c.id)
                .add_columns(*partition_columns())
            )

        query, tables = self.process_input_query(query)
        temp_tables.extend(t.name for t in tables)
        udf_table = self.create_udf_table(_query)
        temp_tables.append(udf_table.name)
        self.populate_udf_table(udf_table, query)
        q, cols = self.create_result_query(udf_table, query)

        return step_result(q, cols)


@frozen
class UDFSignal(UDF):
    is_generator = False

    def create_udf_table(self, query: Select) -> "Table":
        udf_output_columns: list[sqlalchemy.Column[Any]] = [
            sqlalchemy.Column(col_name, col_type)
            for (col_name, col_type) in self.udf.output.items()
        ]

        return self.catalog.warehouse.create_udf_table(
            self.udf_table_name(), udf_output_columns
        )

    def create_pre_udf_table(self, query: Select) -> "Table":
        columns = [
            sqlalchemy.Column(c.name, c.type)
            for c in query.selected_columns
            if c.name != "id"
        ]
        table = self.catalog.warehouse.create_udf_table(self.udf_table_name(), columns)
        select_q = query.with_only_columns(
            *[c for c in query.selected_columns if c.name != "id"]
        )

        # if there is order by clause we need row_number to preserve order
        # if there is no order by clause we still need row_number to generate
        # unique ids as uniqueness is important for this table
        select_q = select_q.add_columns(
            f.row_number().over(order_by=select_q._order_by_clauses).label("id")
        )

        self.catalog.warehouse.db.execute(
            table.insert().from_select(list(select_q.selected_columns), select_q)
        )
        return table

    def process_input_query(self, query: Select) -> tuple[Select, list["Table"]]:
        if os.getenv("DATACHAIN_DISABLE_QUERY_CACHE", "") not in ("", "0"):
            return query, []
        table = self.create_pre_udf_table(query)
        q: Select = sqlalchemy.select(*table.c)
        if query._order_by_clauses:
            # we are adding ordering only if it's explicitly added by user in
            # query part before adding signals
            q = q.order_by(table.c.id)
        return q, [table]

    def create_result_query(
        self, udf_table, query
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        subq = query.subquery()
        original_cols = [c for c in subq.c if c.name not in partition_col_names]

        # new signal columns that are added to udf_table
        signal_cols = [c for c in udf_table.c if c.name != "id"]
        signal_name_cols = {c.name: c for c in signal_cols}
        cols = signal_cols

        def q(*columns):
            cols1 = []
            cols2 = []
            for c in columns:
                if c.name in partition_col_names:
                    continue
                cols.append(signal_name_cols.get(c.name, c))
                if c.name in signal_name_cols:
                    cols2.append(c)
                else:
                    cols1.append(c)

            if cols2:
                res = (
                    sqlalchemy.select(*cols1)
                    .select_from(subq)
                    .outerjoin(udf_table, udf_table.c.id == subq.c.id)
                    .add_columns(*cols2)
                )
            else:
                res = sqlalchemy.select(*cols1).select_from(subq)

            if query._order_by_clauses:
                # if ordering is used in query part before adding signals, we
                # will have it as order by id from select from pre-created udf table
                res = res.order_by(subq.c.id)

            if self.partition_by is not None:
                subquery = res.subquery()
                res = sqlalchemy.select(*subquery.c).select_from(subquery)

            return res

        return q, [*original_cols, *cols]


@frozen
class RowGenerator(UDF):
    """Extend dataset with new rows."""

    is_generator = True

    def create_udf_table(self, query: Select) -> "Table":
        warehouse = self.catalog.warehouse

        table_name = self.udf_table_name()
        columns: tuple[Column, ...] = tuple(
            Column(name, typ) for name, typ in self.udf.output.items()
        )
        return warehouse.create_dataset_rows_table(
            table_name,
            columns=columns,
            if_not_exists=False,
        )

    def create_result_query(
        self, udf_table, query: Select
    ) -> tuple[QueryGeneratorFunc, list["sqlalchemy.Column"]]:
        if not query._order_by_clauses:
            # if we are not selecting all rows in UDF, we need to ensure that
            # we get the same rows as we got as inputs of UDF since selecting
            # without ordering can be non deterministic in some databases
            c = query.selected_columns
            query = query.order_by(c.id)

        udf_table_query = udf_table.select().subquery()
        udf_table_cols: list[sqlalchemy.Label[Any]] = [
            label(c.name, c) for c in udf_table_query.columns
        ]

        def q(*columns):
            names = {c.name for c in columns}
            # Columns for the generated table.
            cols = [c for c in udf_table_cols if c.name in names]
            return sqlalchemy.select(*cols).select_from(udf_table_query)

        return q, udf_table_query.columns


@frozen
class SQLClause(Step, ABC):
    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        query = query_generator.select()
        new_query = self.apply_sql_clause(query)

        def q(*columns):
            return new_query.with_only_columns(*columns)

        return step_result(q, new_query.selected_columns)

    @abstractmethod
    def apply_sql_clause(self, query):
        pass


@frozen
class SQLSelect(SQLClause):
    args: tuple[Union[str, ColumnElement], ...]

    def apply_sql_clause(self, query) -> Select:
        subquery = query.subquery()

        args = [subquery.c[str(c)] if isinstance(c, (str, C)) else c for c in self.args]
        if not args:
            args = subquery.c

        return sqlalchemy.select(*args).select_from(subquery)


@frozen
class SQLSelectExcept(SQLClause):
    args: tuple[str, ...]

    def apply_sql_clause(self, query: Select) -> Select:
        subquery = query.subquery()
        names = set(self.args)
        args = [c for c in subquery.c if c.name not in names]
        return sqlalchemy.select(*args).select_from(subquery)


@frozen
class SQLMutate(SQLClause):
    args: tuple[ColumnElement, ...]

    def apply_sql_clause(self, query: Select) -> Select:
        subquery = query.subquery()
        return sqlalchemy.select(*subquery.c, *self.args).select_from(subquery)


@frozen
class SQLFilter(SQLClause):
    expressions: tuple[ColumnElement, ...]

    def __and__(self, other):
        return self.__class__(self.expressions + other)

    def apply_sql_clause(self, query: Select) -> Select:
        return query.filter(*self.expressions)


@frozen
class SQLOrderBy(SQLClause):
    args: tuple[ColumnElement, ...]

    def apply_sql_clause(self, query: Select) -> Select:
        return query.order_by(*self.args)


@frozen
class SQLLimit(SQLClause):
    n: int

    def apply_sql_clause(self, query: Select) -> Select:
        return query.limit(self.n)


@frozen
class SQLOffset(SQLClause):
    offset: int

    def apply_sql_clause(self, query: "GenerativeSelect"):
        return query.offset(self.offset)


@frozen
class SQLCount(SQLClause):
    def apply_sql_clause(self, query):
        return sqlalchemy.select(f.count(1)).select_from(query.subquery())


@frozen
class SQLUnion(Step):
    query1: "DatasetQuery"
    query2: "DatasetQuery"

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        q1 = self.query1.apply_steps().select().subquery()
        temp_tables.extend(self.query1.temp_table_names)
        q2 = self.query2.apply_steps().select().subquery()
        temp_tables.extend(self.query2.temp_table_names)
        columns1, columns2 = fill_columns(q1.columns, q2.columns)

        def q(*columns):
            names = {c.name for c in columns}
            col1 = [c for c in columns1 if c.name in names]
            col2 = [c for c in columns2 if c.name in names]
            res = (
                sqlalchemy.select(*col1)
                .select_from(q1)
                .union_all(sqlalchemy.select(*col2).select_from(q2))
            )

            subquery = res.subquery()
            return sqlalchemy.select(*subquery.c).select_from(subquery)

        return step_result(
            q,
            columns1,
            dependencies=self.query1.dependencies | self.query2.dependencies,
        )


@frozen
class SQLJoin(Step):
    query1: "DatasetQuery"
    query2: "DatasetQuery"
    predicates: Union[JoinPredicateType, tuple[JoinPredicateType, ...]]
    inner: bool
    rname: str

    def validate_expression(self, exp: "ClauseElement", q1, q2):
        """
        Checking if columns used in expression actually exist in left / right
        part of the join.
        """
        for c in exp.get_children():
            if isinstance(c, ColumnClause):
                assert isinstance(c.table, TableClause)

                q1_c = q1.c.get(c.name)
                q2_c = q2.c.get(c.name)

                if c.table.name == q1.name and q1_c is None:
                    raise ValueError(
                        f"Column {c.name} was not found in left part of the join"
                    )

                if c.table.name == q2.name and q2_c is None:
                    raise ValueError(
                        f"Column {c.name} was not found in right part of the join"
                    )
                if c.table.name not in [q1.name, q2.name]:
                    raise ValueError(
                        f"Column {c.name} was not found in left or right"
                        " part of the join"
                    )
                continue
            self.validate_expression(c, q1, q2)

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        q1 = self.query1.apply_steps().select().subquery(self.query1.table.name)
        temp_tables.extend(self.query1.temp_table_names)
        q2 = self.query2.apply_steps().select().subquery(self.query2.table.name)
        temp_tables.extend(self.query2.temp_table_names)

        q1_columns = list(q1.c)
        q1_column_names = {c.name for c in q1_columns}
        q2_columns = [
            c
            if c.name not in q1_column_names and c.name != "id"
            else c.label(self.rname.format(name=c.name))
            for c in q2.c
        ]

        res_columns = q1_columns + q2_columns
        predicates = (
            (self.predicates,)
            if not isinstance(self.predicates, tuple)
            else self.predicates
        )

        expressions = []
        for p in predicates:
            if isinstance(p, ColumnClause):
                expressions.append(self.query1.c(p.name) == self.query2.c(p.name))
            elif isinstance(p, str):
                expressions.append(self.query1.c(p) == self.query2.c(p))
            elif isinstance(p, ColumnElement):
                expressions.append(p)
            else:
                raise TypeError(f"Unsupported predicate {p} for join expression")

        if not expressions:
            raise ValueError("Missing predicates")

        join_expression = sqlalchemy.and_(*expressions)
        self.validate_expression(join_expression, q1, q2)

        def q(*columns):
            join_query = sqlalchemy.join(
                q1,
                q2,
                join_expression,
                isouter=not self.inner,
            )

            res = sqlalchemy.select(*columns).select_from(join_query)
            subquery = res.subquery()
            return sqlalchemy.select(*subquery.c).select_from(subquery)

        return step_result(
            q,
            res_columns,
            dependencies=self.query1.dependencies | self.query2.dependencies,
        )


@frozen
class GroupBy(Step):
    """Group rows by a specific column."""

    cols: PartitionByType

    def clone(self) -> "Self":
        return self.__class__(self.cols)

    def apply(
        self, query_generator: QueryGenerator, temp_tables: list[str]
    ) -> StepResult:
        query = query_generator.select()
        grouped_query = query.group_by(*self.cols)

        def q(*columns):
            return grouped_query.with_only_columns(*columns)

        return step_result(q, grouped_query.selected_columns)


def fill_columns(
    *column_iterables: Iterable[ColumnElement],
) -> list[list[ColumnElement]]:
    column_dicts = [{c.name: c for c in columns} for columns in column_iterables]
    combined_columns = {n: c for col_dict in column_dicts for n, c in col_dict.items()}

    result: list[list[ColumnElement]] = [[] for _ in column_dicts]
    for n in combined_columns:
        col = next(col_dict[n] for col_dict in column_dicts if n in col_dict)
        for col_dict, out in zip(column_dicts, result):
            if n in col_dict:
                out.append(col_dict[n])
            else:
                # Cast the NULL to ensure all columns are aware of their type
                # Label it to ensure it's aware of its name
                out.append(sqlalchemy.cast(sqlalchemy.null(), col.type).label(n))
    return result


@attrs.define
class ResultIter:
    _row_iter: Iterable[Any]
    columns: list[str]

    def __iter__(self):
        yield from self._row_iter


class DatasetQuery:
    def __init__(
        self,
        path: str = "",
        name: str = "",
        version: Optional[int] = None,
        catalog: Optional["Catalog"] = None,
        client_config=None,
        recursive: Optional[bool] = True,
        session: Optional[Session] = None,
        anon: bool = False,
        indexing_feature_schema: Optional[dict] = None,
        indexing_column_types: Optional[dict[str, Any]] = None,
    ):
        if client_config is None:
            client_config = {}

        if anon:
            client_config["anon"] = True

        self.steps: list[Step] = []
        self.catalog = catalog or get_catalog(client_config=client_config)
        self._chunk_index: Optional[int] = None
        self._chunk_total: Optional[int] = None
        self.temp_table_names: list[str] = []
        self.dependencies: set[DatasetDependencyType] = set()
        self.table = self.get_table()
        self.starting_step: StartingStep
        self.name: Optional[str] = None
        self.version: Optional[int] = None
        self.feature_schema: Optional[dict] = None
        self.column_types: Optional[dict[str, Any]] = None
        self.session = Session.get(session, catalog=catalog)

        if path:
            self.starting_step = IndexingStep(path, self.catalog, {}, recursive)
            self.feature_schema = indexing_feature_schema
            self.column_types = indexing_column_types
        elif name:
            ds = self.catalog.get_dataset(name)
            self.version = version or ds.latest_version
            self.feature_schema = ds.get_version(self.version).feature_schema
            self.column_types = copy(ds.schema)
            if "id" in self.column_types:
                self.column_types.pop("id")
            self.starting_step = QueryStep(self.catalog, name, self.version)
            # attaching to specific dataset
            self.name = name
            self.version = version
        else:
            raise ValueError("must provide path or name")

    @staticmethod
    def is_storage_path(path):
        return bool(re.compile(r"^[a-zA-Z0-9]+://").match(path))

    def __iter__(self):
        return iter(self.results())

    def __or__(self, other):
        return self.union(other)

    @staticmethod
    def get_table() -> "TableClause":
        table_name = "".join(
            random.choice(string.ascii_letters)  # noqa: S311
            for _ in range(16)
        )
        return sqlalchemy.table(table_name)

    @staticmethod
    def delete(
        name: str, version: Optional[int] = None, catalog: Optional["Catalog"] = None
    ) -> None:
        catalog = catalog or get_catalog()
        version = version or catalog.get_dataset(name).latest_version
        catalog.remove_dataset(name, version)

    @property
    def attached(self) -> bool:
        """
        DatasetQuery is considered "attached" to underlying dataset if it represents
        it completely. If this is the case, name and version of underlying dataset
        will be defined.
        DatasetQuery instance can become attached in two scenarios:
            1. ds = DatasetQuery(name="dogs", version=1) -> ds is attached to dogs
            2. ds = ds.save("dogs", version=1) -> ds is attached to dogs dataset
        It can move to detached state if filter or similar methods are called on it,
        as then it no longer 100% represents underlying datasets.
        """
        return self.name is not None and self.version is not None

    def c(self, name: Union[C, str]) -> "ColumnClause[Any]":
        col = sqlalchemy.column(name) if isinstance(name, str) else name
        col.table = self.table
        return col

    def apply_steps(self) -> QueryGenerator:
        """
        Apply the steps in the query and return the resulting
        sqlalchemy.SelectBase.
        """
        query = self.clone()

        index = os.getenv("DATACHAIN_QUERY_CHUNK_INDEX", self._chunk_index)
        total = os.getenv("DATACHAIN_QUERY_CHUNK_TOTAL", self._chunk_total)

        if index is not None and total is not None:
            index, total = int(index), int(total)  # os.getenv returns str

            if not (0 <= index < total):
                raise ValueError("chunk index must be between 0 and total")

            # Respect limit in chunks
            query.steps = self._chunk_limit(query.steps, index, total)

            # Prepend the chunk filter to the step chain.
            query = query.filter(C.random % total == index)
            query.steps = query.steps[-1:] + query.steps[:-1]

        result = query.starting_step.apply()
        group_by = None
        self.dependencies.update(result.dependencies)

        for step in query.steps:
            if isinstance(step, GroupBy):
                if group_by is not None:
                    raise TypeError("only one group_by allowed")
                group_by = step
                continue

            result = step.apply(
                result.query_generator, self.temp_table_names
            )  # a chain of steps linked by results
            self.dependencies.update(result.dependencies)

        if group_by:
            result = group_by.apply(result.query_generator, self.temp_table_names)
            self.dependencies.update(result.dependencies)

        return result.query_generator

    @staticmethod
    def _chunk_limit(steps: list["Step"], index: int, total: int) -> list["Step"]:
        no_limit_steps = []
        limit = None
        for step in steps:
            # Remember last limit
            if isinstance(step, SQLLimit):
                limit = step.n
            # Only keep non-limit steps
            else:
                no_limit_steps.append(step)
        # Chunk the limit
        if limit:
            limit_modulo = limit % total
            limit = limit // total
            if index < limit_modulo:
                limit += 1
            return [*no_limit_steps, SQLLimit(limit)]
        return steps

    def cleanup(self) -> None:
        """Cleanup any temporary tables."""
        if not self.temp_table_names:
            # Nothing to clean up.
            return
        # This is needed to always use a new connection with all metastore and warehouse
        # implementations, as errors may close or render unusable the existing
        # connections.
        metastore = self.catalog.metastore.clone(use_new_connection=True)
        metastore.cleanup_temp_tables(self.temp_table_names)
        metastore.close()
        warehouse = self.catalog.warehouse.clone(use_new_connection=True)
        warehouse.cleanup_temp_tables(self.temp_table_names)
        warehouse.close()
        self.temp_table_names = []

    def results(self, row_factory=None, **kwargs):
        with self.as_iterable(**kwargs) as result:
            if row_factory:
                cols = result.columns
                return [row_factory(cols, r) for r in result]
            return list(result)

    @contextlib.contextmanager
    def as_iterable(self, **kwargs) -> Iterator[ResultIter]:
        try:
            query = self.apply_steps().select()
            selected_columns = [c.name for c in query.columns]
            yield ResultIter(
                self.catalog.warehouse.dataset_rows_select(query, **kwargs),
                selected_columns,
            )
        finally:
            self.cleanup()

    def extract(
        self, *params: UDFParamSpec, workers=ASYNC_WORKERS, **kwargs
    ) -> Iterable[tuple]:
        """
        Extract columns from each row in the query.

        Returns an iterable of tuples matching the given params.

        To ensure prompt resource cleanup, it is recommended to wrap this
        with contextlib.closing().
        """
        actual_params = [normalize_param(p) for p in params]
        try:
            query = self.apply_steps().select()

            def row_iter() -> Generator[RowDict, None, None]:
                # warehouse isn't threadsafe, we need to clone() it
                # in the thread that uses the results
                warehouse = None
                try:
                    warehouse = self.catalog.warehouse.clone()
                    gen = warehouse.dataset_select_paginated(
                        query, limit=query._limit, order_by=query._order_by_clauses
                    )
                    with contextlib.closing(gen) as rows:
                        yield from rows
                finally:
                    # clone doesn't necessarily create a new connection
                    # we can't do `warehouse.close()` for now. It is a bad design
                    # in clone / close interface that needs to be fixed.
                    pass

            async def get_params(row: RowDict) -> tuple:
                return tuple(
                    [
                        await p.get_value_async(self.catalog, row, mapper, **kwargs)
                        for p in actual_params
                    ]
                )

            MapperCls = OrderedMapper if query._order_by_clauses else AsyncMapper  # noqa: N806
            with contextlib.closing(row_iter()) as rows:
                mapper = MapperCls(get_params, rows, workers=workers)
                yield from mapper.iterate()
        finally:
            self.cleanup()

    def to_records(self) -> list[dict]:
        with self.as_iterable() as result:
            cols = result.columns
            return [dict(zip(cols, row)) for row in result]

    def to_pandas(self) -> "pd.DataFrame":
        records = self.to_records()
        df = pd.DataFrame.from_records(records)
        df.columns = [c.replace(DEFAULT_DELIMITER, ".") for c in df.columns]
        return df

    def shuffle(self) -> "Self":
        # ToDo: implement shaffle based on seed and/or generating random column
        return self.order_by(C.random)

    def sample(self, n) -> "Self":
        """
        Return a random sample from the dataset.

        Args:
            n (int): Number of samples to draw.

        NOTE: Sampled are not deterministic, and streamed/paginated queries or
        multiple workers will draw samples with replacement.
        """
        sampled = self.order_by(rand())

        return sampled.limit(n)

    def show(self, limit=20) -> None:
        df = self.limit(limit).to_pandas()

        options = ["display.max_colwidth", 50, "display.show_dimensions", False]
        with pd.option_context(*options):
            if inside_notebook():
                from IPython.display import display

                display(df)

            else:
                print(df.to_string())

        if len(df) == limit:
            print(f"[limited by {limit} objects]")

    def clone(self, new_table=True) -> "Self":
        obj = copy(self)
        obj.steps = obj.steps.copy()
        if new_table:
            obj.table = self.get_table()
        return obj

    @detach
    def select(self, *args, **kwargs) -> "Self":
        """
        Select the given columns or expressions using a subquery.

        If used with no arguments, this simply creates a subquery and
        select all columns from it.

        Note that the `save` function expects default dataset columns to
        be present. This function is meant to be followed by a call to
        `results` if used to exclude any default columns.

        Example:
            >>> ds.select(C.name, C.size * 10).results()
            >>> ds.select(C.name, size10x=C.size * 10).order_by(C.size10x).results()
        """
        named_args = [v.label(k) for k, v in kwargs.items()]
        query = self.clone()
        query.steps.append(SQLSelect((*args, *named_args)))
        return query

    @detach
    def select_except(self, *args) -> "Self":
        """
        Exclude certain columns from this query using a subquery.

        Note that the `save` function expects default dataset columns to
        be present. This function is meant to be followed by a call to
        `results` if used to exclude any default columns.

        Example:
            >>> (
            ...     ds.mutate(size10x=C.size * 10)
            ...     .order_by(C.size10x)
            ...     .select_except(C.size10x)
            ...     .results()
            ... )
        """

        if not args:
            raise TypeError("select_except expected at least 1 argument, got 0")
        query_args = [c if isinstance(c, str) else c.name for c in args]
        query = self.clone()
        query.steps.append(SQLSelectExcept(query_args))  # type: ignore [arg-type]
        return query

    @detach
    def mutate(self, *args, **kwargs) -> "Self":
        """
        Add new columns to this query.

        This function selects all existing columns from this query and
        adds in the new columns specified.

        Example:
            >>> ds.mutate(size10x=C.size * 10).order_by(C.size10x).results()
        """
        query_args = [v.label(k) for k, v in dict(args, **kwargs).items()]
        query = self.clone()
        query.steps.append(SQLMutate((*query_args,)))
        return query

    @detach
    def filter(self, *args) -> "Self":
        query = self.clone(new_table=False)
        steps = query.steps
        if steps and isinstance(steps[-1], SQLFilter):
            steps[-1] = steps[-1] & args
        else:
            steps.append(SQLFilter(args))
        return query

    @detach
    def order_by(self, *args) -> "Self":
        query = self.clone(new_table=False)
        query.steps.append(SQLOrderBy(args))
        return query

    @detach
    def limit(self, n: int) -> "Self":
        query = self.clone(new_table=False)
        query.steps.append(SQLLimit(n))
        return query

    @detach
    def offset(self, offset: int) -> "Self":
        query = self.clone(new_table=False)
        query.steps.append(SQLOffset(offset))
        return query

    def count(self) -> int:
        query = self.clone()
        query.steps.append(SQLCount())
        return query.results()[0][0]

    def sum(self, col: ColumnElement):
        query = self.clone()
        query.steps.append(SQLSelect((f.sum(col),)))
        return query.results()[0][0]

    def avg(self, col: ColumnElement):
        query = self.clone()
        query.steps.append(SQLSelect((f.avg(col),)))
        return query.results()[0][0]

    def min(self, col: ColumnElement):
        query = self.clone()
        query.steps.append(SQLSelect((f.min(col),)))
        return query.results()[0][0]

    def max(self, col: ColumnElement):
        query = self.clone()
        query.steps.append(SQLSelect((f.max(col),)))
        return query.results()[0][0]

    @detach
    def group_by(self, *cols: ColumnElement) -> "Self":
        query = self.clone()
        query.steps.append(GroupBy(cols))
        return query

    @detach
    def union(self, dataset_query: "DatasetQuery") -> "Self":
        left = self.clone()
        right = dataset_query.clone()
        new_query = self.clone()
        new_query.steps = [SQLUnion(left, right)]
        return new_query

    @detach
    def join(
        self,
        dataset_query: "DatasetQuery",
        predicates: Union[JoinPredicateType, Sequence[JoinPredicateType]],
        inner=False,
        rname="{name}_right",
    ) -> "Self":
        left = self.clone(new_table=False)
        if self.table.name == dataset_query.table.name:
            # for use case where we join with itself, e.g dogs.join(dogs, "name")
            right = dataset_query.clone(new_table=True)
        else:
            right = dataset_query.clone(new_table=False)

        new_query = self.clone()
        predicates = (
            predicates
            if isinstance(predicates, (str, ColumnClause, ColumnElement))
            else tuple(predicates)
        )
        new_query.steps = [SQLJoin(left, right, predicates, inner, rname)]
        return new_query

    @detach
    def chunk(self, index: int, total: int) -> "Self":
        """Split a query into smaller chunks for e.g. parallelization.
        Example:
            >>> query = DatasetQuery(...)
            >>> chunk_1 = query._chunk(0, 2)
            >>> chunk_2 = query._chunk(1, 2)
        Note:
            Bear in mind that `index` is 0-indexed but `total` isn't.
            Use 0/3, 1/3 and 2/3, not 1/3, 2/3 and 3/3.
        """
        query = self.clone()
        query._chunk_index, query._chunk_total = index, total
        return query

    @detach
    def add_signals(
        self,
        udf: UDFType,
        parallel: Optional[int] = None,
        workers: Union[bool, int] = False,
        min_task_size: Optional[int] = None,
        partition_by: Optional[PartitionByType] = None,
        cache: bool = False,
    ) -> "Self":
        """
        Adds one or more signals based on the results from the provided UDF.

        Parallel can optionally be specified as >= 1 for parallel processing with a
        specific number of processes, or set to -1 for the default of
        the number of CPUs (cores) on the current machine.

        For distributed processing with the appropriate distributed module installed,
        workers can optionally be specified as >= 1 for a specific number of workers,
        or set to True for the default of all nodes in the cluster.
        As well, a custom minimum task size (min_task_size) can be provided to send
        at least that minimum number of rows to each distributed worker, mostly useful
        if there are a very large number of small tasks to process.
        """
        if isinstance(udf, UDFClassWrapper):  # type: ignore[unreachable]
            # This is a bare decorated class, "instantiate" it now.
            udf = udf()  # type: ignore[unreachable]
        query = self.clone()
        query.steps.append(
            UDFSignal(
                udf,
                self.catalog,
                partition_by=partition_by,
                parallel=parallel,
                workers=workers,
                min_task_size=min_task_size,
                cache=cache,
            )
        )
        return query

    @detach
    def subtract(self, dq: "DatasetQuery") -> "Self":
        query = self.clone()
        query.steps.append(Subtract(dq, self.catalog))
        return query

    @detach
    def changed(self, dq: "DatasetQuery") -> "Self":
        query = self.clone()
        query.steps.append(Changed(dq, self.catalog))
        return query

    @detach
    def generate(
        self,
        udf: UDFType,
        parallel: Optional[int] = None,
        workers: Union[bool, int] = False,
        min_task_size: Optional[int] = None,
        partition_by: Optional[PartitionByType] = None,
        cache: bool = False,
    ) -> "Self":
        if isinstance(udf, UDFClassWrapper):  # type: ignore[unreachable]
            # This is a bare decorated class, "instantiate" it now.
            udf = udf()  # type: ignore[unreachable]
        query = self.clone()
        steps = query.steps
        steps.append(
            RowGenerator(
                udf,
                self.catalog,
                partition_by=partition_by,
                parallel=parallel,
                workers=workers,
                min_task_size=min_task_size,
                cache=cache,
            )
        )
        return query

    def _add_dependencies(self, dataset: "DatasetRecord", version: int):
        for dependency in self.dependencies:
            if isinstance(dependency, tuple):
                # dataset dependency
                ds_dependency_name, ds_dependency_version = dependency
                self.catalog.metastore.add_dataset_dependency(
                    dataset.name,
                    version,
                    ds_dependency_name,
                    ds_dependency_version,
                )
            else:
                # storage dependency - its name is a valid StorageURI
                storage = self.catalog.get_storage(dependency)
                self.catalog.metastore.add_storage_dependency(
                    StorageURI(dataset.name),
                    version,
                    storage.uri,
                    storage.timestamp_str,
                )

    def exec(self) -> "Self":
        """Execute the query."""
        try:
            query = self.clone()
            query.apply_steps()
        finally:
            self.cleanup()
        return query

    def save(
        self,
        name: Optional[str] = None,
        version: Optional[int] = None,
        feature_schema: Optional[dict] = None,
        **kwargs,
    ) -> "Self":
        """Save the query as a dataset."""
        try:
            if name and version and self.catalog.get_dataset(name).has_version(version):
                raise RuntimeError(f"Dataset {name} already has version {version}")
        except DatasetNotFoundError:
            pass
        if not name and version:
            raise RuntimeError("Cannot set version for temporary datasets")

        if not name:
            name = self.session.generate_temp_dataset_name()

        try:
            query = self.apply_steps()

            columns = [
                c if isinstance(c, Column) else Column(c.name, c.type)
                for c in query.columns
            ]
            if not [c for c in columns if c.name != "id"]:
                raise RuntimeError(
                    "No columns to save in the query. "
                    "Ensure at least one column (other than 'id') is selected."
                )

            dataset = self.catalog.create_dataset(
                name,
                version=version,
                feature_schema=feature_schema,
                columns=columns,
                **kwargs,
            )
            version = version or dataset.latest_version

            dr = self.catalog.warehouse.dataset_rows(dataset)

            # Exclude the id column and let the db create it to avoid unique
            # constraint violations.
            q = query.exclude(("id",))
            if q._order_by_clauses:
                # ensuring we have id sorted by order by clause if it exists in a query
                q = q.add_columns(
                    f.row_number().over(order_by=q._order_by_clauses).label("id")
                )

            cols = tuple(c.name for c in q.columns)
            insert_q = sqlalchemy.insert(dr.get_table()).from_select(cols, q)
            self.catalog.warehouse.db.execute(insert_q, **kwargs)
            self.catalog.metastore.update_dataset_status(
                dataset, DatasetStatus.COMPLETE, version=version
            )
            self.catalog.update_dataset_version_with_warehouse_info(dataset, version)

            self._add_dependencies(dataset, version)  # type: ignore [arg-type]
        finally:
            self.cleanup()
        return self.__class__(name=name, version=version, catalog=self.catalog)


def _get_output_fd_for_write() -> Union[str, int]:
    handle = os.getenv("DATACHAIN_OUTPUT_FD")
    if not handle:
        return os.devnull

    if os.name != "nt":
        return int(handle)

    import msvcrt

    return msvcrt.open_osfhandle(int(handle), os.O_WRONLY)  # type: ignore[attr-defined]


@attrs.define
class ExecutionResult:
    preview: list[dict] = attrs.field(factory=list)
    dataset: Optional[tuple[str, int]] = None
    metrics: dict[str, Any] = attrs.field(factory=dict)


def _send_result(dataset_query: DatasetQuery) -> None:
    class JSONSerialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            if isinstance(obj, bytes):
                return list(obj[:1024])
            return super().default(obj)

    try:
        preview_args: dict[str, Any] = json.loads(
            os.getenv("DATACHAIN_QUERY_PREVIEW_ARGS", "")
        )
    except ValueError:
        preview_args = {}

    columns = preview_args.get("columns") or []

    preview_query = (
        dataset_query.select(*columns)
        .limit(preview_args.get("limit", 10))
        .offset(preview_args.get("offset", 0))
    )

    dataset: Optional[tuple[str, int]] = None
    if dataset_query.attached:
        assert dataset_query.name, "Dataset name should be provided"
        assert dataset_query.version, "Dataset version should be provided"
        dataset = dataset_query.name, dataset_query.version

    preview = preview_query.to_records()
    result = ExecutionResult(preview, dataset, metrics)
    data = attrs.asdict(result)

    with open(_get_output_fd_for_write(), mode="w") as f:
        json.dump(data, f, cls=JSONSerialize)


def query_wrapper(dataset_query: DatasetQuery) -> DatasetQuery:
    """
    Wrapper function that wraps the last statement of user query script.
    Last statement MUST be instance of DatasetQuery, otherwise script exits with
    error code 10
    """
    if not isinstance(dataset_query, DatasetQuery):
        sys.exit(QUERY_SCRIPT_INVALID_LAST_STATEMENT_EXIT_CODE)

    catalog = dataset_query.catalog
    save = bool(os.getenv("DATACHAIN_QUERY_SAVE"))
    save_as = os.getenv("DATACHAIN_QUERY_SAVE_AS")

    if save_as:
        if dataset_query.attached:
            dataset_name = dataset_query.name
            version = dataset_query.version
            assert dataset_name, "Dataset name should be provided in attached mode"
            assert version, "Dataset version should be provided in attached mode"

            dataset = catalog.get_dataset(dataset_name)

            try:
                target_dataset = catalog.get_dataset(save_as)
            except DatasetNotFoundError:
                target_dataset = None

            if target_dataset:
                dataset = catalog.register_dataset(dataset, version, target_dataset)
            else:
                dataset = catalog.register_new_dataset(dataset, version, save_as)

            dataset_query = DatasetQuery(
                name=dataset.name,
                version=dataset.latest_version,
                catalog=catalog,
            )
        else:
            dataset_query = dataset_query.save(save_as)
    elif save and not dataset_query.attached:
        name = catalog.generate_query_dataset_name()
        dataset_query = dataset_query.save(name)

    _send_result(dataset_query)
    return dataset_query


def _random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)  # noqa: S311
        for i in range(length)
    )


def _feature_predicate(obj):
    return (
        inspect.isclass(obj) and source.isfrommain(obj) and issubclass(obj, BaseModel)
    )


def _imports(obj):
    return not source.isfrommain(obj)


def get_imports(m):
    root = ast.parse(inspect.getsource(m))

    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            module = None
        elif isinstance(node, ast.ImportFrom):
            module = node.module
        else:
            continue

        for n in node.names:
            import_script = ""
            if module:
                import_script += f"from {module} "
            import_script += f"import {n.name}"
            if n.asname:
                import_script += f" as {n.asname}"
            yield import_script
