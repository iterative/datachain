import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from functools import cached_property, wraps
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    Union,
)

import sqlalchemy
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    Table,
    UniqueConstraint,
    exists,
    select,
)
from sqlalchemy.dialects import sqlite
from sqlalchemy.schema import CreateIndex, CreateTable, DropTable
from sqlalchemy.sql import func
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList
from sqlalchemy.sql.expression import bindparam, cast
from sqlalchemy.sql.selectable import Select
from tqdm.auto import tqdm

import datachain.sql.sqlite
from datachain import semver
from datachain.data_storage import AbstractDBMetastore, AbstractWarehouse
from datachain.data_storage.db_engine import DatabaseEngine
from datachain.data_storage.schema import DefaultSchema
from datachain.dataset import DatasetRecord, StorageURI
from datachain.error import DataChainError, OutdatedDatabaseSchemaError
from datachain.namespace import Namespace
from datachain.project import Project
from datachain.sql.sqlite import create_user_defined_sql_functions, sqlite_dialect
from datachain.sql.sqlite.base import load_usearch_extension
from datachain.sql.types import SQLType
from datachain.utils import DataChainDir, batched_it

if TYPE_CHECKING:
    from sqlalchemy.dialects.sqlite import Insert
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.schema import SchemaItem
    from sqlalchemy.sql._typing import _FromClauseArgument, _OnClauseArgument
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.types import TypeEngine

    from datachain.lib.file import File


logger = logging.getLogger("datachain")

RETRY_START_SEC = 0.01
RETRY_MAX_TIMES = 10
RETRY_FACTOR = 2

DETECT_TYPES = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

datachain.sql.sqlite.setup()

quote_schema = sqlite_dialect.identifier_preparer.quote_schema
quote = sqlite_dialect.identifier_preparer.quote

# NOTE! This should be manually increased when we change our DB schema in codebase
SCHEMA_VERSION = 1

OUTDATED_SCHEMA_ERROR_MESSAGE = (
    "You have an old version of the database schema. Please refer to the documentation"
    " for more information."
)


def _get_in_memory_uri():
    return "file::memory:?cache=shared"


def get_retry_sleep_sec(retry_count: int) -> int:
    return RETRY_START_SEC * (RETRY_FACTOR**retry_count)


def retry_sqlite_locks(func):
    # This retries the database modification in case of concurrent access
    @wraps(func)
    def wrapper(*args, **kwargs):
        exc = None
        for retry_count in range(RETRY_MAX_TIMES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as operror:
                exc = operror
                sleep(get_retry_sleep_sec(retry_count))
        raise exc

    return wrapper


def get_db_file_in_memory(
    db_file: Optional[str] = None, in_memory: bool = False
) -> Optional[str]:
    """Get in-memory db_file and check that conflicting arguments are not provided."""
    if in_memory:
        if db_file and db_file != ":memory:":
            raise RuntimeError("A db_file cannot be specified if in_memory is True")
        db_file = ":memory:"
    return db_file


class SQLiteDatabaseEngine(DatabaseEngine):
    dialect = sqlite_dialect

    db: sqlite3.Connection
    db_file: Optional[str]
    is_closed: bool

    def __init__(
        self,
        engine: "Engine",
        metadata: "MetaData",
        db: sqlite3.Connection,
        db_file: Optional[str] = None,
        max_variable_number: Optional[int] = 999,
    ):
        self.engine = engine
        self.metadata = metadata
        self.db = db
        self.db_file = db_file
        self.is_closed = False
        self.max_variable_number = max_variable_number

    @classmethod
    def from_db_file(cls, db_file: Optional[str] = None) -> "SQLiteDatabaseEngine":
        return cls(*cls._connect(db_file=db_file))

    @staticmethod
    def _connect(
        db_file: Optional[str] = None,
    ) -> tuple["Engine", "MetaData", sqlite3.Connection, str, int]:
        try:
            if db_file == ":memory:":
                # Enable multithreaded usage of the same in-memory db
                db = sqlite3.connect(
                    _get_in_memory_uri(), uri=True, detect_types=DETECT_TYPES
                )
            else:
                db_file = db_file or DataChainDir.find().db
                db = sqlite3.connect(db_file, detect_types=DETECT_TYPES)
            create_user_defined_sql_functions(db)
            engine = sqlalchemy.create_engine(
                "sqlite+pysqlite:///", creator=lambda: db, future=True
            )
            # ensure we run SA on_connect init (e.g it registers regexp function),
            # also makes sure that it's consistent. Otherwise in some cases it
            # seems we are getting different results if engine object is used in a
            # different thread first and enine is not used in the Main thread.
            engine.connect().close()

            db.isolation_level = None  # Use autocommit mode
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA cache_size = -102400")  # 100 MiB
            # Enable Write-Ahead Log Journaling
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = NORMAL")
            db.execute("PRAGMA case_sensitive_like = ON")

            max_variable_number = 999  # minimum in old SQLite versions
            for row in db.execute("PRAGMA compile_options;").fetchall():
                option = row[0]
                if option.startswith("MAX_VARIABLE_NUMBER="):
                    max_variable_number = int(option.split("=")[1])

            if os.environ.get("DEBUG_SHOW_SQL_QUERIES"):
                import sys

                db.set_trace_callback(lambda stmt: print(stmt, file=sys.stderr))

            load_usearch_extension(db)

            return engine, MetaData(), db, db_file, max_variable_number
        except RuntimeError:
            raise DataChainError("Can't connect to SQLite DB") from None

    def clone(self) -> "SQLiteDatabaseEngine":
        """Clones DatabaseEngine implementation."""
        return SQLiteDatabaseEngine.from_db_file(self.db_file)

    def clone_params(self) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
        """
        Returns the function, args, and kwargs needed to instantiate a cloned copy
        of this DatabaseEngine implementation, for use in separate processes
        or machines.
        """
        return (
            SQLiteDatabaseEngine.from_db_file,
            [self.db_file],
            {},
        )

    def _reconnect(self) -> None:
        if not self.is_closed:
            raise RuntimeError("Cannot reconnect on still-open DB!")
        engine, metadata, db, db_file, max_variable_number = self._connect(
            db_file=self.db_file
        )
        self.engine = engine
        self.metadata = metadata
        self.db = db
        self.db_file = db_file
        self.max_variable_number = max_variable_number
        self.is_closed = False

    def get_table(self, name: str) -> Table:
        if self.is_closed:
            # Reconnect in case of being closed previously.
            self._reconnect()
        return super().get_table(name)

    @retry_sqlite_locks
    def execute(
        self,
        query,
        cursor: Optional[sqlite3.Cursor] = None,
        conn=None,
    ) -> sqlite3.Cursor:
        if self.is_closed:
            # Reconnect in case of being closed previously.
            self._reconnect()
        if cursor is not None:
            result = cursor.execute(*self.compile_to_args(query))
        elif conn is not None:
            result = conn.execute(*self.compile_to_args(query))
        else:
            result = self.db.execute(*self.compile_to_args(query))
        if isinstance(query, CreateTable) and query.element.indexes:
            for index in query.element.indexes:
                self.execute(CreateIndex(index, if_not_exists=True), cursor=cursor)
        return result

    @retry_sqlite_locks
    def executemany(
        self, query, params, cursor: Optional[sqlite3.Cursor] = None, conn=None
    ) -> sqlite3.Cursor:
        if cursor:
            return cursor.executemany(self.compile(query).string, params)
        if conn:
            return conn.executemany(self.compile(query).string, params)
        return self.db.executemany(self.compile(query).string, params)

    @retry_sqlite_locks
    def execute_str(self, sql: str, parameters=None) -> sqlite3.Cursor:
        if parameters is None:
            return self.db.execute(sql)
        return self.db.execute(sql, parameters)

    def insert_dataframe(self, table_name: str, df) -> int:
        # Dynamically calculates chunksize by dividing max variable limit in a
        # single SQL insert with number of columns in dataframe.
        # This way we avoid error: sqlite3.OperationalError: too many SQL variables,
        num_columns = df.shape[1]
        if num_columns == 0:
            num_columns = 1

        if self.max_variable_number < num_columns:
            raise RuntimeError(
                "Number of columns exceeds DB maximum variables when inserting data"
            )

        chunksize = self.max_variable_number // num_columns

        return df.to_sql(
            table_name,
            self.db,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )

    def cursor(self, factory=None):
        if factory is None:
            return self.db.cursor()
        return self.db.cursor(factory)

    def close(self) -> None:
        self.db.close()
        self.is_closed = True

    @contextmanager
    def transaction(self):
        db = self.db
        with db:
            db.execute("begin")
            yield db

    def has_table(self, name: str) -> bool:
        """
        Return True if a table exists with the given name

        We cannot simply use `inspect(engine).has_table(name)` like the
        parent class does because that will return False for a table
        created during a pending transaction. Instead, we check the
        sqlite_master table.
        """
        query = select(
            exists(
                select(1)
                .select_from(sqlalchemy.table("sqlite_master"))
                .where(
                    (sqlalchemy.column("type") == "table")
                    & (sqlalchemy.column("name") == name)
                )
            )
        )
        return bool(next(self.execute(query))[0])

    @property
    def table_names(self) -> list[str]:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        return [r[0] for r in self.execute_str(query).fetchall()]

    def create_table(self, table: "Table", if_not_exists: bool = True) -> None:
        self.execute(CreateTable(table, if_not_exists=if_not_exists))

    def drop_table(self, table: "Table", if_exists: bool = False) -> None:
        self.execute(DropTable(table, if_exists=if_exists))

    def rename_table(self, old_name: str, new_name: str):
        comp_old_name = quote_schema(old_name)
        comp_new_name = quote_schema(new_name)
        self.execute_str(f"ALTER TABLE {comp_old_name} RENAME TO {comp_new_name}")


class SQLiteMetastore(AbstractDBMetastore):
    """
    SQLite Metastore uses SQLite3 for storing indexed data locally.
    This is currently used for the local cli.
    """

    META_TABLE = "meta"

    db: "SQLiteDatabaseEngine"

    def __init__(
        self,
        uri: Optional[StorageURI] = None,
        db: Optional["SQLiteDatabaseEngine"] = None,
        db_file: Optional[str] = None,
        in_memory: bool = False,
    ):
        uri = uri or StorageURI("")
        self.schema: DefaultSchema = DefaultSchema()
        super().__init__(uri)

        # needed for dropping tables in correct order for tests because of
        # foreign keys
        self.default_table_names: list[str] = []

        db_file = get_db_file_in_memory(db_file, in_memory)

        self.db = db or SQLiteDatabaseEngine.from_db_file(db_file)

        self._init_meta_table()
        self._init_meta_schema_value()
        self._check_schema_version()
        self._init_tables()
        self._init_namespaces_projects()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close connection upon exit from context manager."""
        self.close()

    def clone(
        self,
        uri: Optional[StorageURI] = None,
        use_new_connection: bool = False,
    ) -> "SQLiteMetastore":
        uri = uri or StorageURI("")
        if not uri and self.uri:
            uri = self.uri

        return SQLiteMetastore(uri=uri, db=self.db.clone())

    def clone_params(self) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
        """
        Returns the class, args, and kwargs needed to instantiate a cloned copy of this
        SQLiteDataStorage implementation, for use in separate processes or machines.
        """
        return (
            SQLiteMetastore.init_after_clone,
            [],
            {
                "uri": self.uri,
                "db_clone_params": self.db.clone_params(),
            },
        )

    @classmethod
    def init_after_clone(
        cls,
        *,
        uri: StorageURI,
        db_clone_params: tuple[Callable, list, dict[str, Any]],
    ) -> "SQLiteMetastore":
        (db_class, db_args, db_kwargs) = db_clone_params
        return cls(uri=uri, db=db_class(*db_args, **db_kwargs))

    @cached_property
    def _meta(self) -> Table:
        return Table(self.META_TABLE, self.db.metadata, *self._meta_columns())

    def _meta_select(self, *columns) -> "Select":
        if not columns:
            return self._meta.select()
        return select(*columns)

    def _meta_insert(self) -> "Insert":
        return sqlite.insert(self._meta)

    def _init_meta_table(self) -> None:
        """Initializes meta table"""
        # NOTE! needs to be called before _init_tables()
        table_names = self.db.table_names
        if table_names and self.META_TABLE not in table_names:
            # this will happen on first run
            raise OutdatedDatabaseSchemaError(OUTDATED_SCHEMA_ERROR_MESSAGE)

        self.db.create_table(self._meta, if_not_exists=True)
        self.default_table_names.append(self._meta.name)

    def _init_meta_schema_value(self) -> None:
        """Inserts current schema version value if not present in meta table yet"""
        stmt = (
            self._meta_insert()
            .values(id=1, schema_version=SCHEMA_VERSION)
            .on_conflict_do_nothing(index_elements=["id"])
        )
        self.db.execute(stmt)

    def _init_tables(self) -> None:
        """Initialize tables."""
        self.db.create_table(self._namespaces, if_not_exists=True)
        self.default_table_names.append(self._namespaces.name)
        self.db.create_table(self._projects, if_not_exists=True)
        self.default_table_names.append(self._projects.name)
        self.db.create_table(self._datasets, if_not_exists=True)
        self.default_table_names.append(self._datasets.name)
        self.db.create_table(self._datasets_versions, if_not_exists=True)
        self.default_table_names.append(self._datasets_versions.name)
        self.db.create_table(self._datasets_dependencies, if_not_exists=True)
        self.default_table_names.append(self._datasets_dependencies.name)
        self.db.create_table(self._jobs, if_not_exists=True)
        self.default_table_names.append(self._jobs.name)

    def _init_namespaces_projects(self) -> None:
        """
        Creates local namespace and local project connected to it.
        In local environment user cannot explicitly create other namespaces and
        projects and all datasets user creates will be stored in those.
        When pulling dataset from Studio, then other namespaces and projects will
        be created implicitly though, to keep the same fully qualified name with
        Studio dataset.
        """
        system_namespace = self.create_namespace(
            Namespace.system(), "System namespace", validate=False
        )
        self.create_project(
            system_namespace.name, Project.listing(), "Listing project", validate=False
        )

    def _check_schema_version(self) -> None:
        """
        Checks if current DB schema is up to date with latest DB model and schema
        version. If not, OutdatedDatabaseSchemaError is raised.
        """
        schema_version = next(self.db.execute(self._meta_select()))[1]
        if schema_version < SCHEMA_VERSION:
            raise OutdatedDatabaseSchemaError(OUTDATED_SCHEMA_ERROR_MESSAGE)

    #
    # Dataset dependencies
    #
    @classmethod
    def _meta_columns(cls) -> list["SchemaItem"]:
        return [
            Column("id", Integer, primary_key=True),
            Column("schema_version", Integer, default=SCHEMA_VERSION),
        ]

    @classmethod
    def _datasets_columns(cls) -> list["SchemaItem"]:
        """Datasets table columns."""
        return [*super()._datasets_columns(), UniqueConstraint("project_id", "name")]

    @classmethod
    def _namespaces_columns(cls) -> list["SchemaItem"]:
        """Datasets table columns."""
        return [*super()._namespaces_columns(), UniqueConstraint("name")]

    def _namespaces_insert(self) -> "Insert":
        return sqlite.insert(self._namespaces)

    def _projects_insert(self) -> "Insert":
        return sqlite.insert(self._projects)

    def _datasets_insert(self) -> "Insert":
        return sqlite.insert(self._datasets)

    def _datasets_versions_insert(self) -> "Insert":
        return sqlite.insert(self._datasets_versions)

    def _datasets_dependencies_insert(self) -> "Insert":
        return sqlite.insert(self._datasets_dependencies)

    #
    # Dataset dependencies
    #

    def _dataset_dependencies_select_columns(self) -> list["SchemaItem"]:
        return [
            self._namespaces.c.name,
            self._projects.c.name,
            self._datasets_dependencies.c.id,
            self._datasets_dependencies.c.dataset_id,
            self._datasets_dependencies.c.dataset_version_id,
            self._datasets.c.name,
            self._datasets_versions.c.version,
            self._datasets_versions.c.created_at,
        ]

    #
    # Jobs
    #

    def _jobs_insert(self) -> "Insert":
        return sqlite.insert(self._jobs)

    @property
    def is_studio(self) -> bool:
        return False

    #
    # Namespaces
    #

    @property
    def default_namespace_name(self):
        return Namespace.default()

    #
    # Projects
    #

    @property
    def default_project_name(self):
        return Project.default()


class SQLiteWarehouse(AbstractWarehouse):
    """
    SQLite Warehouse uses SQLite3 for storing indexed data locally.
    This is currently used for the local cli.
    """

    db: "SQLiteDatabaseEngine"

    # Cache for our defined column types to dialect specific TypeEngine relations
    _col_python_type: ClassVar[dict[type, "TypeEngine"]] = {}

    def __init__(
        self,
        db: Optional["SQLiteDatabaseEngine"] = None,
        db_file: Optional[str] = None,
        in_memory: bool = False,
    ):
        self.schema: DefaultSchema = DefaultSchema()
        super().__init__()

        db_file = get_db_file_in_memory(db_file, in_memory)

        self.db = db or SQLiteDatabaseEngine.from_db_file(db_file)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close connection upon exit from context manager."""
        self.close()

    def clone(self, use_new_connection: bool = False) -> "SQLiteWarehouse":
        return SQLiteWarehouse(db=self.db.clone())

    def clone_params(self) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
        """
        Returns the class, args, and kwargs needed to instantiate a cloned copy of this
        SQLiteDataStorage implementation, for use in separate processes or machines.
        """
        return (
            SQLiteWarehouse.init_after_clone,
            [],
            {"db_clone_params": self.db.clone_params()},
        )

    @classmethod
    def init_after_clone(
        cls,
        *,
        db_clone_params: tuple[Callable, list, dict[str, Any]],
    ) -> "SQLiteWarehouse":
        (db_class, db_args, db_kwargs) = db_clone_params
        return cls(db=db_class(*db_args, **db_kwargs))

    def _reflect_tables(self, filter_tables=None):
        """
        Since some tables are prone to schema extension, meaning we can add
        additional columns to it, we should reflect changes in metadata
        to have the latest columns when dealing with those tables.
        If filter function is defined, it's used to filter out tables to reflect,
        otherwise all tables are reflected
        """
        self.db.metadata.reflect(
            bind=self.db.engine,
            extend_existing=True,
            only=filter_tables,
        )

    def is_ready(self, timeout: Optional[int] = None) -> bool:
        return True

    def create_dataset_rows_table(
        self,
        name: str,
        columns: Sequence["sqlalchemy.Column"] = (),
        if_not_exists: bool = True,
    ) -> Table:
        table = self.schema.dataset_row_cls.new_table(
            name,
            columns=columns,
            metadata=self.db.metadata,
        )
        self.db.create_table(table, if_not_exists=if_not_exists)
        return table

    def get_dataset_sources(
        self, dataset: DatasetRecord, version: str
    ) -> list[StorageURI]:
        dr = self.dataset_rows(dataset, version)
        query = dr.select(dr.c("source", column="file")).distinct()
        cur = self.db.cursor()
        cur.row_factory = sqlite3.Row  # type: ignore[assignment]

        return [
            StorageURI(row["file__source"])
            for row in self.db.execute(query, cursor=cur)
        ]

    def merge_dataset_rows(
        self,
        src: DatasetRecord,
        dst: DatasetRecord,
        src_version: str,
        dst_version: str,
    ) -> None:
        dst_empty = False

        if not self.db.has_table(self.dataset_table_name(src, src_version)):
            # source table doesn't exist, nothing to do
            return

        src_dr = self.dataset_rows(src, src_version).table

        if not self.db.has_table(self.dataset_table_name(dst, dst_version)):
            # destination table doesn't exist, create it
            self.create_dataset_rows_table(
                self.dataset_table_name(dst, dst_version),
                columns=src_dr.columns,
            )
            dst_empty = True

        dst_dr = self.dataset_rows(dst, dst_version).table
        merge_fields = [c.name for c in src_dr.columns if c.name != "sys__id"]
        select_src = select(*(getattr(src_dr.columns, f) for f in merge_fields))

        if dst_empty:
            # we don't need union, but just select from source to destination
            insert_query = sqlite.insert(dst_dr).from_select(merge_fields, select_src)
        else:
            dst_version_latest = None
            # find the previous version of the destination dataset
            dst_previous_versions = [
                v.version
                for v in dst.versions  # type: ignore [union-attr]
                if semver.compare(v.version, dst_version) == -1
            ]
            if dst_previous_versions:
                dst_version_latest = max(dst_previous_versions)

            dst_dr_latest = self.dataset_rows(dst, dst_version_latest).table

            select_dst_latest = select(
                *(getattr(dst_dr_latest.c, f) for f in merge_fields)
            )
            union_query = sqlalchemy.union(select_src, select_dst_latest)
            insert_query = (
                sqlite.insert(dst_dr)
                .from_select(merge_fields, union_query)
                .prefix_with("OR IGNORE")
            )

        self.db.execute(insert_query)

    def prepare_entries(self, entries: "Iterable[File]") -> Iterable[dict[str, Any]]:
        return (e.model_dump() for e in entries)

    def insert_rows(self, table: Table, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return

        with self.db.transaction() as conn:
            # transactions speeds up inserts significantly as there is no separate
            # transaction created for each insert row
            self.db.executemany(
                table.insert().values({f: bindparam(f) for f in rows[0]}),
                rows,
                conn=conn,
            )

    def insert_dataset_rows(self, df, dataset: DatasetRecord, version: str) -> int:
        dr = self.dataset_rows(dataset, version)
        return self.db.insert_dataframe(dr.table.name, df)

    def instr(self, source, target) -> "ColumnElement":
        return cast(func.instr(source, target), sqlalchemy.Boolean)

    def get_table(self, name: str) -> sqlalchemy.Table:
        # load table with latest schema to metadata
        self._reflect_tables(filter_tables=lambda t, _: t == name)
        return self.db.metadata.tables[name]

    def python_type(self, col_type: Union["TypeEngine", "SQLType"]) -> Any:
        if isinstance(col_type, SQLType):
            # converting our defined column types to dialect specific TypeEngine
            col_type_cls = type(col_type)
            if col_type_cls not in self._col_python_type:
                self._col_python_type[col_type_cls] = col_type.type_engine(
                    sqlite_dialect
                )
            col_type = self._col_python_type[col_type_cls]

        return col_type.python_type

    def dataset_table_export_file_names(
        self, dataset: DatasetRecord, version: str
    ) -> list[str]:
        raise NotImplementedError("Exporting dataset table not implemented for SQLite")

    def export_dataset_table(
        self,
        bucket_uri: str,
        dataset: DatasetRecord,
        version: str,
        client_config=None,
    ) -> list[str]:
        raise NotImplementedError("Exporting dataset table not implemented for SQLite")

    def copy_table(
        self,
        table: Table,
        query: Select,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        col_id = (
            query.selected_columns.sys__id
            if "sys__id" in query.selected_columns
            else None
        )

        # If there is no sys__id column, we cannot copy the table in batches,
        # and we need to copy all rows at once. Same if there is a group by clause.
        if col_id is None or len(query._group_by_clause) > 0:
            select_q = query.with_only_columns(
                *[c for c in query.selected_columns if c.name != "sys__id"]
            )
            q = table.insert().from_select(list(select_q.selected_columns), select_q)
            self.db.execute(q)
            return

        select_ids = query.with_only_columns(col_id)
        ids = self.db.execute(select_ids).fetchall()

        select_q = (
            query.with_only_columns(
                *[c for c in query.selected_columns if c.name != "sys__id"]
            )
            .offset(None)
            .limit(None)
        )

        for batch in batched_it(ids, 10_000):
            batch_ids = [row[0] for row in batch]
            select_q._where_criteria = (col_id.in_(batch_ids),)
            q = table.insert().from_select(list(select_q.selected_columns), select_q)

            self.db.execute(q)

            if progress_cb:
                progress_cb(len(batch_ids))

    def join(
        self,
        left: "_FromClauseArgument",
        right: "_FromClauseArgument",
        onclause: "_OnClauseArgument",
        inner: bool = True,
        full: bool = False,
        columns=None,
    ) -> "Select":
        """
        Join two tables together.
        """
        if not full:
            join_query = sqlalchemy.join(
                left,
                right,
                onclause,
                isouter=not inner,
            )
            return sqlalchemy.select(*columns).select_from(join_query)

        left_right_join = sqlalchemy.select(*columns).select_from(
            sqlalchemy.join(left, right, onclause, isouter=True)
        )
        right_left_join = sqlalchemy.select(*columns).select_from(
            sqlalchemy.join(right, left, onclause, isouter=True)
        )

        def add_left_rows_filter(exp: BinaryExpression):
            """
            Adds filter to right_left_join to remove unmatched left table rows by
            getting column names that need to be NULL from BinaryExpressions in onclause
            """
            return right_left_join.where(
                getattr(left.c, exp.left.name) == None  # type: ignore[union-attr] # noqa: E711
            )

        if isinstance(onclause, BinaryExpression):
            right_left_join = add_left_rows_filter(onclause)

        if isinstance(onclause, BooleanClauseList):
            for c in onclause.get_children():
                if isinstance(c, BinaryExpression):
                    right_left_join = add_left_rows_filter(c)

        union = sqlalchemy.union(left_right_join, right_left_join).subquery()
        return sqlalchemy.select(*union.c).select_from(union)

    def create_pre_udf_table(self, query: "Select") -> "Table":
        """
        Create a temporary table from a query for use in a UDF.
        """
        columns = [
            sqlalchemy.Column(c.name, c.type)
            for c in query.selected_columns
            if c.name != "sys__id"
        ]
        table = self.create_udf_table(columns)

        with tqdm(desc="Preparing", unit=" rows", leave=False) as pbar:
            self.copy_table(table, query, progress_cb=pbar.update)

        return table
