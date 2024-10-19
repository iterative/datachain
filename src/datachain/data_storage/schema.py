import inspect
from collections.abc import Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    TypeVar,
)

import sqlalchemy as sa
from sqlalchemy.sql import func as f
from sqlalchemy.sql.expression import false, null, true

from datachain.sql.functions import path
from datachain.sql.types import Int, SQLType, UInt64

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.engine.interfaces import Dialect
    from sqlalchemy.sql.base import (
        ColumnCollection,
        Executable,
        ReadOnlyColumnCollection,
    )
    from sqlalchemy.sql.elements import ColumnElement


DEFAULT_DELIMITER = "__"


def dedup_columns(columns: Iterable[sa.Column]) -> list[sa.Column]:
    """
    Removes duplicate columns from a list of columns.
    If column with the same name and different type is found, exception is
    raised
    """
    c_set: dict[str, sa.Column] = {}
    for c in columns:
        if (ec := c_set.get(c.name, None)) is not None:
            if str(ec.type) != str(c.type):
                raise ValueError(
                    f"conflicting types for column {c.name}:{c.type!s} and {ec.type!s}"
                )
            continue
        c_set[c.name] = c

    return list(c_set.values())


def convert_rows_custom_column_types(
    columns: "ColumnCollection[str, ColumnElement[Any]]",
    rows: Iterator[tuple[Any, ...]],
    dialect: "Dialect",
) -> Iterator[tuple[Any, ...]]:
    """
    This function converts values of rows columns based on their types which are
    defined in columns. We are only converting column values for which types are
    subclasses of our SQLType, as only for those we have converters registered.
    """
    # indexes of SQLType column in a list of columns so that we can skip the rest
    custom_columns_types: list[tuple[int, SQLType]] = [
        (idx, c.type) for idx, c in enumerate(columns) if isinstance(c.type, SQLType)
    ]

    if not custom_columns_types:
        yield from rows

    for row in rows:
        row_list = list(row)
        for idx, t in custom_columns_types:
            row_list[idx] = (
                t.default_value(dialect)
                if row_list[idx] is None
                else t.on_read_convert(row_list[idx], dialect)
            )

        yield tuple(row_list)


class DirExpansion:
    def __init__(self, object_name: str):
        self.object_name = object_name

    def col_name(self, name: str) -> str:
        return f"{self.object_name}{DEFAULT_DELIMITER}{name}"

    def col(self, query, name: str) -> str:
        return getattr(query.c, self.col_name(name))

    def base_select(self, q):
        return sa.select(
            q.c.sys__id,
            false().label(self.col_name("is_dir")),
            self.col(q, "source"),
            self.col(q, "path"),
            self.col(q, "version"),
            self.col(q, "location"),
        )

    def apply_group_by(self, q):
        return (
            sa.select(
                f.min(q.c.sys__id).label("sys__id"),
                self.col(q, "is_dir"),
                self.col(q, "source"),
                self.col(q, "path"),
                self.col(q, "version"),
                f.max(self.col(q, "location")).label(self.col_name("location")),
            )
            .select_from(q)
            .group_by(
                self.col(q, "source"),
                self.col(q, "path"),
                self.col(q, "is_dir"),
                self.col(q, "version"),
            )
            .order_by(
                self.col(q, "source"),
                self.col(q, "path"),
                self.col(q, "is_dir"),
                self.col(q, "version"),
            )
        )

    def query(self, q):
        q = self.base_select(q).cte(recursive=True)
        parent = path.parent(self.col(q, "path"))
        q = q.union_all(
            sa.select(
                sa.literal(-1).label("sys__id"),
                true().label(self.col_name("is_dir")),
                self.col(q, "source"),
                parent.label(self.col_name("path")),
                sa.literal("").label(self.col_name("version")),
                null().label(self.col_name("location")),
            ).where(parent != "")
        )
        return self.apply_group_by(q)


class DataTable:
    def __init__(
        self,
        name: str,
        engine: "Engine",
        metadata: Optional["sa.MetaData"] = None,
        column_types: Optional[dict[str, SQLType]] = None,
        object_name: str = "file",
    ):
        self.name: str = name
        self.engine = engine
        self.metadata: sa.MetaData = metadata if metadata is not None else sa.MetaData()
        self.column_types: dict[str, SQLType] = column_types or {}
        self.object_name = object_name

    @staticmethod
    def copy_column(
        column: sa.Column,
        primary_key: Optional[bool] = None,
        index: Optional[bool] = None,
        nullable: Optional[bool] = None,
        default: Optional[Any] = None,
        server_default: Optional[Any] = None,
        unique: Optional[bool] = None,
    ) -> sa.Column:
        """
        Copy a sqlalchemy Column object intended for use as a signal column.

        This does not copy all attributes as certain attributes such as
        table are too context-dependent and the purpose of this function is
        adding a signal column from one table to another table.

        We can't use Column.copy() as it only works in certain contexts.
        See https://github.com/sqlalchemy/sqlalchemy/issues/5953
        """
        return sa.Column(
            column.name,
            column.type,
            primary_key=primary_key if primary_key is not None else column.primary_key,
            index=index if index is not None else column.index,
            nullable=nullable if nullable is not None else column.nullable,
            default=default if default is not None else column.default,
            server_default=(
                server_default if server_default is not None else column.server_default
            ),
            unique=unique if unique is not None else column.unique,
        )

    @classmethod
    def new_table(
        cls,
        name: str,
        columns: Sequence["sa.Column"] = (),
        metadata: Optional["sa.MetaData"] = None,
    ):
        # copy columns, since re-using the same objects from another table
        # may raise an error
        columns = cls.sys_columns() + [cls.copy_column(c) for c in columns]
        columns = dedup_columns(columns)

        if metadata is None:
            metadata = sa.MetaData()
        return sa.Table(name, metadata, *columns)

    def get_table(self) -> "sa.Table":
        table = self.metadata.tables.get(self.name)
        if table is None:
            sa.Table(self.name, self.metadata, autoload_with=self.engine)
            # ^^^ This table may not be correctly initialised on some dialects
            # Grab it from metadata instead.
            table = self.metadata.tables[self.name]

        column_types = self.column_types | {c.name: c.type for c in self.sys_columns()}
        # adjusting types for custom columns to be instances of SQLType if possible
        for c in table.columns:
            if c.name in column_types:
                t = column_types[c.name]
                c.type = t() if inspect.isclass(t) else t
        return table

    @property
    def columns(self) -> "ReadOnlyColumnCollection[str, sa.Column[Any]]":
        return self.table.columns

    @property
    def c(self):
        return self.columns

    def col_name(self, name: str) -> str:
        return f"{self.object_name}{DEFAULT_DELIMITER}{name}"

    def col(self, name: str):
        return getattr(self.c, self.col_name(name))

    @property
    def table(self) -> "sa.Table":
        return self.get_table()

    @staticmethod
    def to_db_name(name: str) -> str:
        return name.replace(".", DEFAULT_DELIMITER)

    def apply_conditions(self, query: "Executable") -> "Executable":
        """
        Apply any conditions that belong on all selecting queries.

        This could be used to filter tables that use access control.
        """
        return query

    def select(self, *columns):
        if not columns:
            query = self.table.select()
        else:
            query = sa.select(*columns)
        return self.apply_conditions(query)

    def insert(self):
        return self.table.insert()

    def update(self):
        return self.apply_conditions(self.table.update())

    def delete(self):
        return self.apply_conditions(self.table.delete())

    @staticmethod
    def sys_columns():
        return [
            sa.Column("sys__id", Int, primary_key=True),
            sa.Column(
                "sys__rand", UInt64, nullable=False, server_default=f.abs(f.random())
            ),
        ]

    def dir_expansion(self):
        return DirExpansion(self.object_name)
        # return DirExpansion(self.object_name).query(query)


PARTITION_COLUMN_ID = "partition_id"

partition_col_names = [PARTITION_COLUMN_ID]


def partition_columns() -> Sequence["sa.Column"]:
    return [
        sa.Column(PARTITION_COLUMN_ID, sa.Integer),
    ]


DataTableT = TypeVar("DataTableT", bound=DataTable)


class Schema(Generic[DataTableT]):
    dataset_row_cls: type[DataTableT]


class DefaultSchema(Schema[DataTable]):
    def __init__(self):
        self.dataset_row_cls = DataTable
