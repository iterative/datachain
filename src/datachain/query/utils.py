from typing import Optional, Union

import sqlalchemy as sa

ColT = Union[sa.ColumnClause, sa.Column, sa.ColumnElement, sa.TextClause, sa.Label]


def column_name(col: ColT) -> str:
    """Returns column name from column element."""
    if isinstance(col, (sa.TextClause)):
        raise TypeError(f"Got text clause column in select: {col!s}")
    return col.name


def get_query_column(query: sa.Select, name: str) -> Optional[ColT]:
    """Returns column element from query by name or None if column not found."""
    return next((col for col in query.inner_columns if column_name(col) == name), None)


def get_query_id_column(query: sa.Select) -> Optional[sa.ColumnElement]:
    """Returns ID column element from query or None if column not found."""
    col = get_query_column(query, "sys__id")
    return col if col is not None and isinstance(col, sa.ColumnElement) else None
