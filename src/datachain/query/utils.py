from typing import Optional, Union

import sqlalchemy as sa

ColT = Union[sa.Column, sa.ColumnElement, sa.TextClause]


def column_name(col: ColT) -> str:
    """Returns column name from column element."""
    return col.name if isinstance(col, sa.Column) else str(col)


def get_query_column(query: sa.Select, name: str) -> Optional[ColT]:
    """Returns column element from query by name or None if column not found."""
    return next((col for col in query.inner_columns if column_name(col) == name), None)


def get_query_id_column(query: sa.Select) -> Optional[sa.ColumnElement]:
    """Returns ID column element from query or None if column not found."""
    col = get_query_column(query, "sys__id")
    return col if col is not None and isinstance(col, sa.ColumnElement) else None


def select_only_columns(query: sa.Select, *names: str) -> sa.Select:
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
