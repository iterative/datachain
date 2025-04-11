import contextlib
import itertools
import os
import sqlite3
from typing import TYPE_CHECKING, Any, Optional, Union

import sqlalchemy

if TYPE_CHECKING:
    from datachain.lib.dc.utils import OutputType
    from datachain.query import Session

    from .datachain import DataChain


def read_database(
    query: Union[str, "sqlalchemy.TextClause", "sqlalchemy.Selectable"],
    connection: Union[
        "sqlalchemy.Connectable", str, "sqlalchemy.URL", "sqlite3.Connection"
    ],
    session: Optional["Session"] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    ds_name: str = "",
    *,
    output: Optional["OutputType"] = None,
    execute_options: Optional[dict[str, Any]] = None,
) -> "DataChain":
    """
    Generate chain from the database query.

    Args:
        query: SQL query to execute.
        connection : SQLAlchemy connectable, str, or a sqlite3 connection
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported. The user is
            responsible for engine disposal and connection closure for the
            SQLAlchemy connectable; str connections are closed automatically.

    Example:
        ```py
        import datachain as dc

        query = "SELECT key, value FROM table"
        connection = "sqlite:///example.db"
        dc.read_database(query, connection)
        ```
    """

    from datachain.lib.convert.values_to_tuples import values_to_tuples
    from datachain.lib.dc.records import read_records

    with contextlib.ExitStack() as stack:
        engine_kwargs = {"echo": bool(os.environ.get("DEBUG_SHOW_SQL_QUERIES"))}
        if isinstance(connection, (str, sqlalchemy.URL)):
            engine = sqlalchemy.create_engine(connection, **engine_kwargs)
            stack.callback(engine.dispose)
            connection = stack.enter_context(engine.connect())
        elif isinstance(connection, sqlite3.Connection):
            engine = sqlalchemy.create_engine(
                "sqlite+pysqlite:///", creator=lambda: connection, **engine_kwargs
            )
            connection = engine.connect()

        assert isinstance(connection, sqlalchemy.Connection)

        execute_options = execute_options or {}
        execute_options.setdefault("stream_results", True)  # use server-side cursors
        if isinstance(query, str):
            query = sqlalchemy.text(query)

        result = connection.execute(query, execution_options=execute_options)  # type: ignore[arg-type]
        result = stack.enter_context(result)
        # TODO: improve schema inference
        # Also, how to support nullable types?
        if first_row := result.fetchone():
            _, output, _ = values_to_tuples(
                ds_name,
                output,
                **{col: [v] for col, v in first_row._mapping.items()},
            )
            result = itertools.chain([first_row], result)  # type: ignore[assignment]

        # TODO: How to make this lazy
        return read_records(
            (row._asdict() for row in result),
            session=session,
            settings=settings,
            in_memory=in_memory,
            schema=output,  # type: ignore[arg-type]
        )
