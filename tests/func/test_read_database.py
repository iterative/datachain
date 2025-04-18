import os
import sqlite3
from contextlib import closing
from typing import Optional

import pytest
import sqlalchemy
from sqlalchemy.orm import Session

from datachain import read_database
from datachain.lib.dc import database
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def db_path(tmp_dir):
    return tmp_dir / "main.db"


@pytest.fixture
def db_uri(db_path):
    return "sqlite:///" + os.fspath(db_path)


@pytest.fixture
def db_engine(db_uri):
    engine = sqlalchemy.create_engine(db_uri)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_connection(db_engine):
    with closing(db_engine.connect()) as conn:
        yield conn


@pytest.fixture
def db_session(db_engine):
    with Session(bind=db_engine) as session:
        yield session


@pytest.fixture
def sqlite3_connection(db_path):
    with sqlite3.connect(db_path) as conn:
        yield conn


@pytest.fixture(
    params=(
        "db_uri",
        "db_connection",
        "db_engine",
        "db_session",
        "sqlite3_connection",
    )
)
def connection(request):
    return request.getfixturevalue(request.param)


def test(sqlite3_connection, connection, test_session):
    sqlite3_connection.execute("CREATE TABLE tbl (id INTEGER PRIMARY KEY, value TEXT)")
    sqlite3_connection.executemany(
        "INSERT INTO tbl(value) VALUES(?)", [(str(i),) for i in range(1, 1000)]
    )
    sqlite3_connection.commit()

    chain = read_database(
        "select * from tbl where id > :val",
        connection,
        params={"val": 25},
        session=test_session,
    )
    assert chain.schema == {"id": int, "value": str}
    assert sorted(chain.to_records(), key=lambda r: r["id"]) == [
        {"id": i, "value": str(i)} for i in range(26, 1000)
    ]


# FIXME: `clickhouse` requires wrapping column types in `Nullable` to make the column
# nullable, setting `nullable=True` is not enough.
# https://github.com/xzkostyan/clickhouse-sqlalchemy/issues/189#issuecomment-1274736713
# Also, was not able to figure out how to read nullable columns back from clickhouse.


@skip_if_not_sqlite
def test_nullable(sqlite3_connection, test_session):
    """
    Verify that a column containing a sequence of NULL values is handled correctly
    when the number of leading NULLs is less than `infer_schema_length`.
    """
    sqlite3_connection.execute("CREATE TABLE tbl (id INTEGER PRIMARY KEY, value TEXT)")
    sqlite3_connection.executemany(
        "INSERT INTO tbl(value) VALUES(?)",
        [(None if i < 50 else str(i),) for i in range(1, 1000)],
    )
    sqlite3_connection.commit()

    chain = read_database("select * from tbl", sqlite3_connection, session=test_session)
    assert chain.schema == {"id": int, "value": Optional[str]}
    assert sorted(chain.to_records(), key=lambda r: r["id"]) == [
        {"id": i, "value": None if i < 50 else str(i)} for i in range(1, 1000)
    ]


@skip_if_not_sqlite
def test_all_null_values(sqlite3_connection, test_session):
    sqlite3_connection.execute("CREATE TABLE tbl (id INTEGER PRIMARY KEY, num INTEGER)")
    sqlite3_connection.executemany(
        "INSERT INTO tbl(num) VALUES(?)", [(None,) for _ in range(1, 1000)]
    )
    sqlite3_connection.commit()

    chain = read_database("select * from tbl", sqlite3_connection, session=test_session)
    # if all values are null, the column type defaults to str
    assert chain.schema == {"id": int, "num": Optional[str]}
    assert sorted(chain.to_records(), key=lambda r: r["id"]) == [
        {"id": i, "num": None} for i in range(1, 1000)
    ]


def test_empty(sqlite3_connection, test_session):
    sqlite3_connection.execute("CREATE TABLE tbl (id INTEGER PRIMARY KEY, value TEXT)")

    chain = read_database("select * from tbl", sqlite3_connection, session=test_session)
    # if the table is empty, the column type defaults to str
    assert chain.schema == {"id": Optional[str], "value": Optional[str]}
    assert chain.to_records() == []


def test_overriding_schema_with_output(sqlite3_connection, test_session):
    sqlite3_connection.execute(
        "CREATE TABLE tbl (id INTEGER PRIMARY KEY, value INTEGER)"
    )
    sqlite3_connection.executemany(
        "INSERT INTO tbl(value) VALUES(?)", [(i,) for i in range(1, 1000)]
    )
    sqlite3_connection.commit()

    chain = read_database(
        "select * from tbl",
        sqlite3_connection,
        output={"value": float},
        session=test_session,
    )
    assert chain.schema == {"id": int, "value": float}
    assert sorted(chain.to_records(), key=lambda r: r["id"]) == [
        {"id": i, "value": float(i)} for i in range(1, 1000)
    ]


def test_schema_is_not_inferred_when_all_types_are_provided(
    mocker, sqlite3_connection, test_session
):
    sqlite3_connection.execute(
        "CREATE TABLE tbl (id INTEGER PRIMARY KEY, value INTEGER)"
    )
    sqlite3_connection.executemany(
        "INSERT INTO tbl(value) VALUES(?)", [(i,) for i in range(1, 10)]
    )
    sqlite3_connection.commit()

    spy = mocker.spy(database, "_infer_schema")
    chain = read_database(
        "select * from tbl",
        sqlite3_connection,
        output={"id": int, "value": int},
        session=test_session,
    )
    spy.assert_called_once_with(mocker.ANY, [], 100)
    assert chain.schema == {"id": int, "value": int}
