import glob
import json
import logging
import posixpath
import random
import string
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from urllib.parse import urlparse

import attrs
import sqlalchemy as sa
from sqlalchemy import Table, case, select
from sqlalchemy.sql import func
from sqlalchemy.sql.expression import true
from tqdm import tqdm

from datachain.client import Client
from datachain.data_storage.schema import convert_rows_custom_column_types
from datachain.data_storage.serializer import Serializable
from datachain.dataset import DatasetRecord
from datachain.node import DirType, DirTypeGroup, Entry, Node, NodeWithPath, get_path
from datachain.sql.functions import path as pathfunc
from datachain.sql.types import Int, SQLType
from datachain.storage import StorageURI
from datachain.utils import sql_escape_like

if TYPE_CHECKING:
    from sqlalchemy.sql._typing import _ColumnsClauseArgument
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.selectable import Select
    from sqlalchemy.types import TypeEngine

    from datachain.data_storage import AbstractIDGenerator, schema
    from datachain.data_storage.db_engine import DatabaseEngine
    from datachain.data_storage.schema import DataTable

try:
    import numpy as np

    numpy_imported = True
except ImportError:
    numpy_imported = False


logger = logging.getLogger("datachain")

SELECT_BATCH_SIZE = 100_000  # number of rows to fetch at a time


class AbstractWarehouse(ABC, Serializable):
    """
    Abstract Warehouse class, to be implemented by any Database Adapters
    for a specific database system. This manages the storing, searching, and
    retrieval of datasets data, and has shared logic for all database
    systems currently in use.
    """

    #
    # Constants, Initialization, and Tables
    #

    DATASET_TABLE_PREFIX = "ds_"
    DATASET_SOURCE_TABLE_PREFIX = "src_"
    UDF_TABLE_NAME_PREFIX = "udf_"
    TMP_TABLE_NAME_PREFIX = "tmp_"

    id_generator: "AbstractIDGenerator"
    schema: "schema.Schema"
    db: "DatabaseEngine"

    def __init__(self, id_generator: "AbstractIDGenerator"):
        self.id_generator = id_generator

    def __enter__(self) -> "AbstractWarehouse":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Default behavior is to do nothing, as connections may be shared.
        pass

    def cleanup_for_tests(self):
        """Cleanup for tests."""

    def convert_type(  # noqa: PLR0911
        self,
        val: Any,
        col_type: SQLType,
        col_python_type: Any,
        col_type_name: str,
        col_name: str,
    ) -> Any:
        """
        Tries to convert value to specific type if needed and if compatible,
        otherwise throws an ValueError.
        If value is a list or some other iterable, it tries to convert sub elements
        as well
        """
        if numpy_imported and isinstance(val, (np.ndarray, np.generic)):
            val = val.tolist()

        # Optimization: Precompute all the column type variables.
        value_type = type(val)

        exc = None
        try:
            if col_python_type is list and value_type in (list, tuple, set):
                if len(val) == 0:
                    return []
                item_python_type = self.python_type(col_type.item_type)
                if item_python_type is not list:
                    if isinstance(val[0], item_python_type):
                        return val
                    if item_python_type is float and isinstance(val[0], int):
                        return [float(i) for i in val]
                # Optimization: Reuse these values for each function call within the
                # list comprehension.
                item_type_info = (
                    col_type.item_type,
                    item_python_type,
                    type(col_type.item_type).__name__,
                    col_name,
                )
                return [self.convert_type(i, *item_type_info) for i in val]
            # Special use case with JSON type as we save it as string
            if col_python_type is dict or col_type_name == "JSON":
                if value_type is str:
                    return val
                if value_type in (dict, list):
                    return json.dumps(val)
                raise ValueError(
                    f"Cannot convert value {val!r} with type {value_type} to JSON"
                )

            if isinstance(val, col_python_type):
                return val
            if col_python_type is float and isinstance(val, int):
                return float(val)
        except Exception as e:  # noqa: BLE001
            exc = e
        ve = ValueError(
            f"Value {val!r} with type {value_type} incompatible for "
            f"column type {col_type_name}"
        )
        # This is the same as "from exc" when not raising the exception.
        if exc:
            ve.__cause__ = exc
        # Optimization: Log here, so only one try/except is needed, since the ValueError
        # above is raised after logging.
        logger.exception(
            "Error while validating/converting type for column "
            "%s with value %s, original error %s",
            col_name,
            val,
            ve,
        )
        raise ve

    @abstractmethod
    def clone(self, use_new_connection: bool = False) -> "AbstractWarehouse":
        """Clones Warehouse implementation for some Storage input.
        Setting use_new_connection will always use a new database connection.
        New connections should only be used if needed due to errors with
        closed connections."""

    def close(self) -> None:
        """Closes any active database connections."""
        self.db.close()

    def close_on_exit(self) -> None:
        """Closes any active database or HTTP connections, called on Session exit or
        for test cleanup only, as some Warehouse implementations may handle this
        differently."""
        self.close()

    #
    # Query Tables
    #

    @abstractmethod
    def is_ready(self, timeout: Optional[int] = None) -> bool: ...

    def dataset_rows(self, dataset: DatasetRecord, version: Optional[int] = None):
        version = version or dataset.latest_version

        table_name = self.dataset_table_name(dataset.name, version)
        return self.schema.dataset_row_cls(
            table_name,
            self.db.engine,
            self.db.metadata,
            dataset.get_schema(version),
        )

    @property
    def dataset_row_cls(self) -> type["DataTable"]:
        return self.schema.dataset_row_cls

    #
    # Query Execution
    #

    def dataset_select_paginated(
        self,
        query,
        page_size: int = SELECT_BATCH_SIZE,
    ) -> Generator[Sequence, None, None]:
        """
        This is equivalent to `db.execute`, but for selecting rows in batches
        """
        limit = query._limit
        paginated_query = query.limit(page_size)

        if not paginated_query._order_by_clauses:
            # default order by is order by `sys__id`
            paginated_query = paginated_query.order_by(query.selected_columns.sys__id)

        results = None
        offset = 0
        num_yielded = 0
        try:
            while True:
                if limit is not None:
                    limit -= num_yielded
                    if limit == 0:
                        break
                    if limit < page_size:
                        paginated_query = paginated_query.limit(None).limit(limit)

                results = self.dataset_rows_select(paginated_query.offset(offset))

                processed = False
                for row in results:
                    processed = True
                    yield row
                    num_yielded += 1

                if not processed:
                    break  # no more results
                offset += page_size
        finally:
            # https://www2.sqlite.org/cvstrac/wiki?p=DatabaseIsLocked (SELECT not
            # finalized or reset) to prevent database table is locked error when an
            # exception is raised in the middle of processing the results (e.g.
            # https://github.com/iterative/dvcx/issues/924). Connections close
            # apparently is not enough in some cases, at least on sqlite
            # https://www.sqlite.org/c3ref/close.html
            if results and hasattr(results, "close"):
                results.close()

    #
    # Table Name Internal Functions
    #

    @staticmethod
    def uri_to_storage_info(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        name = parsed.path if parsed.scheme == "file" else parsed.netloc
        return parsed.scheme, name

    def dataset_table_name(self, dataset_name: str, version: int) -> str:
        prefix = self.DATASET_TABLE_PREFIX
        if Client.is_data_source_uri(dataset_name):
            # for datasets that are created for bucket listing we use different prefix
            prefix = self.DATASET_SOURCE_TABLE_PREFIX
        return f"{prefix}{dataset_name}_{version}"

    def temp_table_name(self) -> str:
        return self.TMP_TABLE_NAME_PREFIX + _random_string(6)

    def udf_table_name(self) -> str:
        return self.UDF_TABLE_NAME_PREFIX + _random_string(6)

    #
    # Datasets
    #

    @abstractmethod
    def create_dataset_rows_table(
        self,
        name: str,
        columns: Sequence["sa.Column"] = (),
        if_not_exists: bool = True,
    ) -> Table:
        """Creates a dataset rows table for the given dataset name and columns"""

    def drop_dataset_rows_table(
        self,
        dataset: DatasetRecord,
        version: int,
        if_exists: bool = True,
    ) -> None:
        """Drops a dataset rows table for the given dataset name."""
        table_name = self.dataset_table_name(dataset.name, version)
        table = Table(table_name, self.db.metadata)
        self.db.drop_table(table, if_exists=if_exists)

    @abstractmethod
    def merge_dataset_rows(
        self,
        src: "DatasetRecord",
        dst: "DatasetRecord",
        src_version: int,
        dst_version: int,
    ) -> None:
        """
        Merges source dataset rows and current latest destination dataset rows
        into a new rows table created for new destination dataset version.
        Note that table for new destination version must be created upfront.
        Merge results should not contain duplicates.
        """

    def dataset_rows_select(
        self,
        query: sa.sql.selectable.Select,
        **kwargs,
    ) -> Iterator[tuple[Any, ...]]:
        """
        Fetch dataset rows from database.
        """
        rows = self.db.execute(query, **kwargs)
        yield from convert_rows_custom_column_types(
            query.selected_columns, rows, self.db.dialect
        )

    @abstractmethod
    def get_dataset_sources(
        self, dataset: DatasetRecord, version: int
    ) -> list[StorageURI]: ...

    def nodes_dataset_query(
        self,
        dataset_rows: "DataTable",
        *,
        column_names: Iterable[str],
        path: Optional[str] = None,
        recursive: Optional[bool] = False,
    ) -> "sa.Select":
        """
        Creates query pointing to certain bucket listing represented by dataset_rows
        The given `column_names`
        will be selected in the order they're given. `path` is a glob which
        will select files in matching directories, or if `recursive=True` is
        set then the entire tree under matching directories will be selected.
        """
        dr = dataset_rows

        def _is_glob(path: str) -> bool:
            return any(c in path for c in ["*", "?", "[", "]"])

        column_objects = [dr.c[c] for c in column_names]
        # include all object types - file, tar archive, tar file (subobject)
        select_query = dr.select(*column_objects).where(
            dr.c.dir_type.in_(DirTypeGroup.FILE) & (dr.c.is_latest == true())
        )
        if path is None:
            return select_query
        if recursive:
            root = False
            where = self.path_expr(dr).op("GLOB")(path)
            if not path or path == "/":
                # root of the bucket, e.g s3://bucket/ -> getting all the nodes
                # in the bucket
                root = True

            if not root and not _is_glob(path):
                # not a root and not a explicit glob, so it's pointing to some directory
                # and we are adding a proper glob syntax for it
                # e.g s3://bucket/dir1 -> s3://bucket/dir1/*
                dir_path = path.rstrip("/") + "/*"
                where = where | self.path_expr(dr).op("GLOB")(dir_path)

            if not root:
                # not a root, so running glob query
                select_query = select_query.where(where)

        else:
            parent = self.get_node_by_path(dr, path.lstrip("/").rstrip("/*"))
            select_query = select_query.where(pathfunc.parent(dr.c.path) == parent.path)
        return select_query

    def rename_dataset_table(
        self,
        old_name: str,
        new_name: str,
        old_version: int,
        new_version: int,
    ) -> None:
        old_ds_table_name = self.dataset_table_name(old_name, old_version)
        new_ds_table_name = self.dataset_table_name(new_name, new_version)

        self.db.rename_table(old_ds_table_name, new_ds_table_name)

    def dataset_rows_count(self, dataset: DatasetRecord, version=None) -> int:
        """Returns total number of rows in a dataset"""
        dr = self.dataset_rows(dataset, version)
        table = dr.get_table()
        query = select(sa.func.count(table.c.sys__id))
        (res,) = self.db.execute(query)
        return res[0]

    def dataset_stats(
        self, dataset: DatasetRecord, version: int
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Returns tuple with dataset stats: total number of rows and total dataset size.
        """
        if not (self.db.has_table(self.dataset_table_name(dataset.name, version))):
            return None, None

        dr = self.dataset_rows(dataset, version)
        table = dr.get_table()
        expressions: tuple[_ColumnsClauseArgument[Any], ...] = (
            sa.func.count(table.c.sys__id),
        )
        if "file__size" in table.columns:
            expressions = (*expressions, sa.func.sum(table.c.file__size))
        elif "size" in table.columns:
            expressions = (*expressions, sa.func.sum(table.c.size))
        query = select(*expressions)
        ((nrows, *rest),) = self.db.execute(query)
        return nrows, rest[0] if rest else None

    def prepare_entries(
        self, uri: str, entries: Iterable[Entry]
    ) -> list[dict[str, Any]]:
        """
        Prepares bucket listing entry (row) for inserting into database
        """

        def _prepare_entry(entry: Entry):
            assert entry.dir_type is not None
            return attrs.asdict(entry) | {"source": uri}

        return [_prepare_entry(e) for e in entries]

    @abstractmethod
    def insert_rows(self, table: Table, rows: Iterable[dict[str, Any]]) -> None:
        """Does batch inserts of any kind of rows into table"""

    def insert_rows_done(self, table: Table) -> None:
        """
        Only needed for certain implementations
        to signal when rows inserts are complete.
        """

    @abstractmethod
    def insert_dataset_rows(self, df, dataset: DatasetRecord, version: int) -> int:
        """Inserts dataset rows directly into dataset table"""

    @abstractmethod
    def instr(self, source, target) -> "ColumnElement":
        """
        Return SQLAlchemy Boolean determining if a target substring is present in
        source string column
        """

    @abstractmethod
    def get_table(self, name: str) -> sa.Table:
        """
        Returns a SQLAlchemy Table object by name. If table doesn't exist, it should
        create it
        """

    @abstractmethod
    def dataset_table_export_file_names(
        self, dataset: DatasetRecord, version: int
    ) -> list[str]:
        """
        Returns list of file names that will be created when user runs dataset export
        """

    @abstractmethod
    def export_dataset_table(
        self,
        bucket_uri: str,
        dataset: DatasetRecord,
        version: int,
        client_config=None,
    ) -> list[str]:
        """
        Exports dataset table to the cloud, e.g to some s3 bucket
        """

    def python_type(self, col_type: Union["TypeEngine", "SQLType"]) -> Any:
        """Returns python type representation of some Sqlalchemy column type"""
        return col_type.python_type

    def add_node_type_where(
        self,
        query: sa.Select,
        type: str,
        include_subobjects: bool = True,
    ) -> sa.Select:
        file_group: Sequence[int]
        if type in {"f", "file", "files"}:
            if include_subobjects:
                file_group = DirTypeGroup.SUBOBJ_FILE
            else:
                file_group = DirTypeGroup.FILE
        elif type in {"d", "dir", "directory", "directories"}:
            if include_subobjects:
                file_group = DirTypeGroup.SUBOBJ_DIR
            else:
                file_group = DirTypeGroup.DIR
        else:
            raise ValueError(f"invalid file type: {type!r}")

        c = query.selected_columns
        q = query.where(c.dir_type.in_(file_group))
        if not include_subobjects:
            q = q.where(c.vtype == "")
        return q

    def get_nodes(self, query) -> Iterator[Node]:
        """
        This gets nodes based on the provided query, and should be used sparingly,
        as it will be slow on any OLAP database systems.
        """
        columns = [c.name for c in query.selected_columns]
        for row in self.db.execute(query):
            d = dict(zip(columns, row))
            yield Node(**d)

    def get_dirs_by_parent_path(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
    ) -> Iterator[Node]:
        """Gets nodes from database by parent path, with optional filtering"""
        dr = dataset_rows
        query = self._find_query(
            dr,
            parent_path,
            type="dir",
            conds=[pathfunc.parent(sa.Column("path")) == parent_path],
            order_by=["source", "path"],
        )
        return self.get_nodes(query)

    def _get_nodes_by_glob_path_pattern(
        self, dataset_rows: "DataTable", path_list: list[str], glob_name: str
    ) -> Iterator[Node]:
        """Finds all Nodes that correspond to GLOB like path pattern."""
        dr = dataset_rows
        de = dr.dataset_dir_expansion(
            dr.select().where(dr.c.is_latest == true()).subquery()
        ).subquery()
        path_glob = "/".join([*path_list, glob_name])
        dirpath = path_glob[: -len(glob_name)]
        relpath = func.substr(self.path_expr(de), len(dirpath) + 1)

        return self.get_nodes(
            self.expand_query(de, dr)
            .where(
                (self.path_expr(de).op("GLOB")(path_glob))
                & ~self.instr(relpath, "/")
                & (self.path_expr(de) != dirpath)
            )
            .order_by(de.c.source, de.c.path, de.c.version)
        )

    def _get_node_by_path_list(
        self, dataset_rows: "DataTable", path_list: list[str], name: str
    ) -> Node:
        """
        Gets node that correspond some path list, e.g ["data-lakes", "dogs-and-cats"]
        """
        parent = "/".join(path_list)
        dr = dataset_rows
        de = dr.dataset_dir_expansion(
            dr.select().where(dr.c.is_latest == true()).subquery()
        ).subquery()
        query = self.expand_query(de, dr)

        q = query.where(de.c.path == get_path(parent, name)).order_by(
            de.c.source, de.c.path, de.c.version
        )
        row = next(self.dataset_rows_select(q), None)
        if not row:
            path = f"{parent}/{name}"
            raise FileNotFoundError(f"Unable to resolve path {path!r}")
        return Node(*row)

    def _populate_nodes_by_path(
        self, dataset_rows: "DataTable", path_list: list[str]
    ) -> list[Node]:
        """
        Puts all nodes found by path_list into the res input variable.
        Note that path can have GLOB like pattern matching which means that
        res can have multiple nodes as result.
        If there is no GLOB pattern, res should have one node as result that
        match exact path by path_list
        """
        if not path_list:
            return [self._get_node_by_path_list(dataset_rows, [], "")]
        matched_paths: list[list[str]] = [[]]
        for curr_name in path_list[:-1]:
            if glob.has_magic(curr_name):
                new_paths: list[list[str]] = []
                for path in matched_paths:
                    nodes = self._get_nodes_by_glob_path_pattern(
                        dataset_rows, path, curr_name
                    )
                    new_paths.extend([*path, n.name] for n in nodes if n.is_container)
                matched_paths = new_paths
            else:
                for path in matched_paths:
                    path.append(curr_name)
        curr_name = path_list[-1]
        if glob.has_magic(curr_name):
            result: list[Node] = []
            for path in matched_paths:
                nodes = self._get_nodes_by_glob_path_pattern(
                    dataset_rows, path, curr_name
                )
                result.extend(nodes)
        else:
            result = [
                self._get_node_by_path_list(dataset_rows, path, curr_name)
                for path in matched_paths
            ]
        return result

    @staticmethod
    def expand_query(dir_expanded_query, dataset_rows: "DataTable"):
        dr = dataset_rows
        de = dir_expanded_query

        def with_default(column):
            default = getattr(attrs.fields(Node), column.name).default
            return func.coalesce(column, default).label(column.name)

        return sa.select(
            de.c.sys__id,
            with_default(dr.c.vtype),
            case((de.c.is_dir == true(), DirType.DIR), else_=dr.c.dir_type).label(
                "dir_type"
            ),
            de.c.path,
            with_default(dr.c.etag),
            de.c.version,
            with_default(dr.c.is_latest),
            dr.c.last_modified,
            with_default(dr.c.size),
            with_default(dr.c.owner_name),
            with_default(dr.c.owner_id),
            with_default(dr.c.sys__rand),
            dr.c.location,
            de.c.source,
        ).select_from(de.outerjoin(dr.table, de.c.sys__id == dr.c.sys__id))

    def get_node_by_path(self, dataset_rows: "DataTable", path: str) -> Node:
        """Gets node that corresponds to some path"""
        if path == "":
            return Node.root()
        dr = dataset_rows
        if not path.endswith("/"):
            query = dr.select().where(
                self.path_expr(dr) == path,
                dr.c.is_latest == true(),
                dr.c.dir_type != DirType.DIR,
            )
            row = next(self.db.execute(query), None)
            if row is not None:
                return Node(*row)
            path += "/"
        query = sa.select(1).where(
            dr.select()
            .where(
                dr.c.is_latest == true(),
                dr.c.dir_type != DirType.DIR,
                dr.c.path.startswith(path),
            )
            .exists()
        )
        row = next(self.db.execute(query), None)
        if not row:
            raise FileNotFoundError(f"Unable to resolve path {path}")
        path = path.removesuffix("/")
        return Node.from_dir(path)

    def expand_path(self, dataset_rows: "DataTable", path: str) -> list[Node]:
        """Simulates Unix-like shell expansion"""
        clean_path = path.strip("/")
        path_list = clean_path.split("/") if clean_path != "" else []
        res = self._populate_nodes_by_path(dataset_rows, path_list)
        if path.endswith("/"):
            res = [node for node in res if node.dir_type in DirTypeGroup.SUBOBJ_DIR]
        return res

    def select_node_fields_by_parent_path(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
        fields: Iterable[str],
    ) -> Iterator[tuple[Any, ...]]:
        """
        Gets latest-version file nodes from the provided parent path
        """
        dr = dataset_rows
        de = dr.dataset_dir_expansion(
            dr.select().where(dr.c.is_latest == true()).subquery()
        ).subquery()
        where_cond = pathfunc.parent(de.c.path) == parent_path
        if parent_path == "":
            # Exclude the root dir
            where_cond = where_cond & (de.c.path != "")
        inner_query = self.expand_query(de, dr).where(where_cond).subquery()

        def field_to_expr(f):
            if f == "name":
                return pathfunc.name(inner_query.c.path)
            return getattr(inner_query.c, f)

        return self.db.execute(
            select(*(field_to_expr(f) for f in fields)).order_by(
                inner_query.c.source,
                inner_query.c.path,
                inner_query.c.version,
            )
        )

    def select_node_fields_by_parent_path_tar(
        self, dataset_rows: "DataTable", parent_path: str, fields: Iterable[str]
    ) -> Iterator[tuple[Any, ...]]:
        """
        Gets latest-version file nodes from the provided parent path
        """
        dr = dataset_rows
        dirpath = f"{parent_path}/"

        def field_to_expr(f):
            if f == "name":
                return pathfunc.name(dr.c.path)
            return getattr(dr.c, f)

        q = (
            select(*(field_to_expr(f) for f in fields))
            .where(
                self.path_expr(dr).like(f"{sql_escape_like(dirpath)}%"),
                ~self.instr(pathfunc.name(dr.c.path), "/"),
                dr.c.is_latest == true(),
            )
            .order_by(dr.c.source, dr.c.path, dr.c.version, dr.c.etag)
        )
        return self.db.execute(q)

    def size(
        self,
        dataset_rows: "DataTable",
        node: Union[Node, dict[str, Any]],
        count_files: bool = False,
    ) -> tuple[int, int]:
        """
        Calculates size of some node (and subtree below node).
        Returns size in bytes as int and total files as int
        """
        if isinstance(node, dict):
            is_dir = node.get("is_dir", node["dir_type"] in DirTypeGroup.SUBOBJ_DIR)
            node_size = node["size"]
            path = node["path"]
        else:
            is_dir = node.is_container
            node_size = node.size
            path = node.path
        if not is_dir:
            # Return node size if this is not a directory
            return node_size, 1

        sub_glob = posixpath.join(path, "*")
        dr = dataset_rows
        selections = [
            func.sum(dr.c.size),
        ]
        if count_files:
            selections.append(
                func.sum(dr.c.dir_type.in_(DirTypeGroup.FILE)),
            )
        results = next(
            self.db.execute(
                dr.select(*selections).where(
                    (self.path_expr(dr).op("GLOB")(sub_glob))
                    & (dr.c.is_latest == true())
                )
            ),
            (0, 0),
        )
        if count_files:
            return results[0] or 0, results[1] or 0
        return results[0] or 0, 0

    def path_expr(self, t):
        return t.c.path

    def _find_query(
        self,
        dataset_rows: "DataTable",
        parent_path: str,
        fields: Optional[Sequence[str]] = None,
        type: Optional[str] = None,
        conds=None,
        order_by: Optional[Union[str, list[str]]] = None,
        include_subobjects: bool = True,
    ) -> sa.Select:
        if not conds:
            conds = []

        dr = dataset_rows
        de = dr.dataset_dir_expansion(
            dr.select().where(dr.c.is_latest == true()).subquery()
        ).subquery()
        q = self.expand_query(de, dr).subquery()
        path = self.path_expr(q)

        if parent_path:
            sub_glob = posixpath.join(parent_path, "*")
            conds.append(path.op("GLOB")(sub_glob))
        else:
            conds.append(path != "")

        columns = q.c
        if fields:
            columns = [getattr(columns, f) for f in fields]

        query = sa.select(*columns)
        query = query.where(*conds)
        if type is not None:
            query = self.add_node_type_where(query, type, include_subobjects)
        if order_by is not None:
            if isinstance(order_by, str):
                order_by = [order_by]
            query = query.order_by(*order_by)
        return query

    def get_subtree_files(
        self,
        dataset_rows: "DataTable",
        node: Node,
        sort: Union[list[str], str, None] = None,
        include_subobjects: bool = True,
    ) -> Iterator[NodeWithPath]:
        """
        Returns all file nodes that are "below" some node.
        Nodes can be sorted as well.
        """
        dr = dataset_rows
        query = self._find_query(
            dr,
            node.path,
            type="f",
            include_subobjects=include_subobjects,
        )
        if sort is not None:
            if not isinstance(sort, list):
                sort = [sort]
            query = query.order_by(*(sa.text(s) for s in sort))  # type: ignore [attr-defined]

        prefix_len = len(node.path)

        def make_node_with_path(node: Node) -> NodeWithPath:
            return NodeWithPath(node, node.path[prefix_len:].lstrip("/").split("/"))

        return map(make_node_with_path, self.get_nodes(query))

    def find(
        self,
        dataset_rows: "DataTable",
        node: Node,
        fields: Sequence[str],
        type=None,
        conds=None,
        order_by=None,
    ) -> Iterator[tuple[Any, ...]]:
        """
        Finds nodes that match certain criteria and only looks for latest nodes
        under the passed node.
        """
        query = self._find_query(
            dataset_rows,
            node.path,
            fields=fields,
            type=type,
            conds=conds,
            order_by=order_by,
        )
        return self.db.execute(query)

    def update_node(self, node_id: int, values: dict[str, Any]) -> None:
        # TODO used only in formats which will be deleted
        """Update entry of a specific node in the database."""

    def create_udf_table(
        self,
        columns: Sequence["sa.Column"] = (),
        name: Optional[str] = None,
    ) -> "sa.Table":
        """
        Create a temporary table for storing custom signals generated by a UDF.
        SQLite TEMPORARY tables cannot be directly used as they are process-specific,
        and UDFs are run in other processes when run in parallel.
        """
        tbl = sa.Table(
            name or self.udf_table_name(),
            sa.MetaData(),
            sa.Column("sys__id", Int, primary_key=True),
            *columns,
        )
        self.db.create_table(tbl, if_not_exists=True)
        return tbl

    @abstractmethod
    def copy_table(
        self,
        table: Table,
        query: "Select",
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        Copy the results of a query into a table.
        """

    @abstractmethod
    def create_pre_udf_table(self, query: "Select") -> "Table":
        """
        Create a temporary table from a query for use in a UDF.
        """

    def is_temp_table_name(self, name: str) -> bool:
        """Returns if the given table name refers to a temporary
        or no longer needed table."""
        return name.startswith(
            (self.TMP_TABLE_NAME_PREFIX, self.UDF_TABLE_NAME_PREFIX, "ds_shadow_")
        ) or name.endswith("_shadow")

    def get_temp_table_names(self) -> list[str]:
        return [
            t
            for t in sa.inspect(self.db.engine).get_table_names()
            if self.is_temp_table_name(t)
        ]

    def cleanup_tables(self, names: Iterable[str]) -> None:
        """
        Drop tables passed.

        This should be implemented to ensure that the provided tables
        are cleaned up as soon as they are no longer needed.
        """
        with tqdm(desc="Cleanup", unit=" tables") as pbar:
            for name in names:
                self.db.drop_table(Table(name, self.db.metadata), if_exists=True)
                pbar.update(1)

    def changed_query(
        self,
        source_query: sa.sql.selectable.Select,
        target_query: sa.sql.selectable.Select,
    ) -> sa.sql.selectable.Select:
        sq = source_query.alias("source_query")
        tq = target_query.alias("target_query")

        source_target_join = sa.join(
            sq, tq, (sq.c.source == tq.c.source) & (sq.c.path == tq.c.path)
        )

        return (
            select(*sq.c)
            .select_from(source_target_join)
            .where(
                (sq.c.last_modified > tq.c.last_modified)
                & (sq.c.is_latest == true())
                & (tq.c.is_latest == true())
            )
        )


def _random_string(length: int) -> str:
    return "".join(
        random.choice(string.ascii_letters + string.digits)  # noqa: S311
        for i in range(length)
    )
