from typing import TYPE_CHECKING, Optional, Union

from sqlalchemy import Column

if TYPE_CHECKING:
    from sqlalchemy import ColumnElement, Select, TextClause


ColT = Union[Column, "ColumnElement", "TextClause"]


def column_name(col: ColT) -> str:
    """Returns column name from column element."""
    return col.name if isinstance(col, Column) else str(col)


def get_query_column(query: "Select", name: str) -> Optional[ColT]:
    """Returns column element from query by name or None if column not found."""
    return next((col for col in query.inner_columns if column_name(col) == name), None)


def get_query_id_column(query: "Select") -> ColT:
    """Returns ID column element from query or None if column not found."""
    col = get_query_column(query, "sys__id")
    if col is None:
        raise RuntimeError("sys__id column not found in query")
    return col


def select_only_columns(query: "Select", *names: str) -> "Select":
    """Returns query selecting defined columns only."""
    if not names:
        return query

    cols: list[ColT] = []
    for name in names:
        col = get_query_column(query, name)
        if col is None:
            raise ValueError(f"Column '{name}' not found in query")
        cols.append(col)

    return query.with_only_columns(*cols)
