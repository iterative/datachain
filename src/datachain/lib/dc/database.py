import contextlib
import itertools
import os
import sqlite3
from typing import TYPE_CHECKING, Any, Optional, Union

import sqlalchemy

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    import sqlalchemy.orm  # noqa: TC004

    from datachain.lib.data_model import DataType
    from datachain.query import Session

    from .datachain import DataChain

    ConnectionType = Union[
        str,
        sqlalchemy.engine.URL,
        sqlalchemy.engine.interfaces.Connectable,
        sqlalchemy.engine.Engine,
        sqlalchemy.engine.Connection,
        sqlalchemy.orm.Session,
        sqlite3.Connection,
    ]


@contextlib.contextmanager
def _connect(
    connection: "ConnectionType",
) -> "Iterator[Union[sqlalchemy.engine.Connection, sqlalchemy.orm.Session]]":
    import sqlalchemy.orm

    with contextlib.ExitStack() as stack:
        engine_kwargs = {"echo": bool(os.environ.get("DEBUG_SHOW_SQL_QUERIES"))}
        if isinstance(connection, (str, sqlalchemy.URL)):
            engine = sqlalchemy.create_engine(connection, **engine_kwargs)
            stack.callback(engine.dispose)
            yield stack.enter_context(engine.connect())
        elif isinstance(connection, sqlite3.Connection):
            engine = sqlalchemy.create_engine(
                "sqlite://", creator=lambda: connection, **engine_kwargs
            )
            # do not close the connection, as it is managed by the caller
            yield engine.connect()
        elif isinstance(connection, sqlalchemy.Engine):
            yield stack.enter_context(connection.connect())
        elif isinstance(connection, (sqlalchemy.Connection, sqlalchemy.orm.Session)):
            # do not close the connection, as it is managed by the caller
            yield connection
        else:
            raise TypeError(f"Unsupported connection type: {type(connection).__name__}")


def _infer_schema(
    result: "sqlalchemy.engine.Result",
    to_infer: list[str],
    infer_schema_length: Optional[int] = 100,
) -> tuple[list["sqlalchemy.Row"], dict[str, "DataType"]]:
    from datachain.lib.convert.values_to_tuples import values_to_tuples

    if not to_infer:
        return [], {}

    rows = list(itertools.islice(result, infer_schema_length))
    values = {col: [row._mapping[col] for row in rows] for col in to_infer}
    _, output_schema, _ = values_to_tuples("", **values)
    return rows, output_schema


def read_database(
    query: Union[str, "sqlalchemy.sql.expression.Executable"],
    connection: "ConnectionType",
    params: Union["Sequence[Mapping[str, Any]]", "Mapping[str, Any]", None] = None,
    *,
    output: Optional["dict[str, DataType]"] = None,
    session: Optional["Session"] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    infer_schema_length: Optional[int] = 100,
) -> "DataChain":
    """
    Read the results of a SQL query into a DataChain, using a given database connection.

    Args:
        query:
            The SQL query to execute. Can be a raw SQL string or a SQLAlchemy
            `Executable` object.
        connection: SQLAlchemy connectable, str, or a sqlite3 connection
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported. The user is
            responsible for engine disposal and connection closure for the
            SQLAlchemy connectable; str connections are closed automatically.
        params: Parameters to pass to execute method.
        output: A dictionary mapping column names to types, used to override the
            schema inferred from the query results.
        session: Session to use for the chain.
        settings: Settings to use for the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        infer_schema_length:
            The maximum number of rows to scan for inferring schema.
            If set to `None`, the full data may be scanned.
            The rows used for schema inference are stored in memory,
            so large values can lead to high memory usage.
            Only applies if the `output` parameter is not set for the given column.

    Examples:
        Reading from a SQL query against a user-supplied connection:
        ```python
        query = "SELECT key, value FROM tbl"
        chain = dc.read_database(query, connection, output={"value": float})
        ```

        Load data from a SQLAlchemy driver/engine:
        ```python
        from sqlalchemy import create_engine
        engine = create_engine("postgresql+psycopg://myuser:mypassword@localhost:5432/mydb")
        chain = dc.read_database("select * from tbl", engine)
        ```

        Load data from a parameterized SQLAlchemy query:
        ```python
        query = "SELECT key, value FROM tbl WHERE value > :value"
        dc.read_database(query, engine, params={"value": 50})
        ```

    Notes:
        - This function works with a variety of databases — including,
        but not limited to, SQLite, DuckDB, PostgreSQL, and Snowflake,
        provided the appropriate driver is installed.
        - This call is blocking, and will execute the query and return once the
          results are saved.
    """
    from datachain.lib.dc.records import read_records

    output = output or {}
    if isinstance(query, str):
        query = sqlalchemy.text(query)
    kw = {"execution_options": {"stream_results": True}}  # use server-side cursors
    with _connect(connection) as conn, conn.execute(query, params, **kw) as result:
        cols = result.keys()
        to_infer = [k for k in cols if k not in output]  # preserve the order
        rows, inferred_schema = _infer_schema(result, to_infer, infer_schema_length)
        records = (row._asdict() for row in itertools.chain(rows, result))
        return read_records(
            records,
            session=session,
            settings=settings,
            in_memory=in_memory,
            schema=inferred_schema | output,
        )
