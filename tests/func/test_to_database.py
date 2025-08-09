import sqlite3
from contextlib import closing
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.orm import Session

import datachain as dc


@pytest.fixture
def db_uri():
    return "sqlite:///:memory:"


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
def sqlite3_connection():
    with sqlite3.connect(":memory:") as conn:
        yield conn


@pytest.fixture(
    params=(
        "db_connection",
        "db_engine",
        "db_session",
        "sqlite3_connection",
    )
)
def connection(request):
    return request.getfixturevalue(request.param)


def test_basic_to_database(tmp_dir, connection):
    """Test basic functionality with actual DataChain data."""
    chain = dc.read_values(
        id=[1, 2, 3], name=["Alice", "Bob", "Charlie"], age=[25, 30, 35]
    )

    chain.to_database("users", connection)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM users ORDER BY id")).fetchall()
        assert len(result) == 3
        assert result[0] == (1, "Alice", 25)
        assert result[1] == (2, "Bob", 30)
        assert result[2] == (3, "Charlie", 35)


def test_to_database_with_uri(db_uri):
    """Test basic functionality with URI connection string."""
    chain = dc.read_values(
        id=[1, 2, 3], name=["Alice", "Bob", "Charlie"], age=[25, 30, 35]
    )

    chain.to_database("users", db_uri)

    # URI connections create isolated databases, so we can't verify externally
    # The absence of exceptions indicates success


def test_to_database_with_complex_types(connection, test_session):
    """Test to_database with complex data types including nested structures."""

    class UserProfile(dc.DataModel):
        name: str
        settings: dict
        tags: list[str]

    chain = dc.read_values(
        id=[1, 2, 3],
        profile=[
            UserProfile(
                name="Alice",
                settings={"theme": "dark", "notifications": True},
                tags=["admin", "vip"],
            ),
            UserProfile(
                name="Bob",
                settings={"theme": "light", "notifications": False},
                tags=["user"],
            ),
            UserProfile(
                name="Charlie",
                settings={"theme": "auto", "notifications": True},
                tags=["user", "beta"],
            ),
        ],
        session=test_session,
    )

    chain.to_database("user_profiles", connection)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("SELECT * FROM user_profiles ORDER BY id")
        )
        rows = result.fetchall()

    assert len(rows) == 3

    # Verify the table schema: DataModel fields are flattened with double underscores
    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("PRAGMA table_info(user_profiles)"))
        columns = [row[1] for row in result.fetchall()]

    expected_columns = ["id", "profile__name", "profile__settings", "profile__tags"]
    assert columns == expected_columns

    # Verify a sample row to confirm data structure and JSON serialization
    # Complex types (dict, list) are serialized as JSON strings
    first_row = rows[0]
    assert first_row[0] == 1
    assert first_row[1] == "Alice"
    assert first_row[2] == '{"theme": "dark", "notifications": true}'
    assert first_row[3] == '["admin","vip"]'  # JSON array (no spaces after commas)


def test_to_database_large_dataset(connection, test_session):
    """Test to_database with large amount of data and custom batch size."""
    large_size = 10000
    chain = dc.read_values(
        id=list(range(large_size)),
        value=[f"value_{i}" for i in range(large_size)],
        number=[i * 2.5 for i in range(large_size)],
        session=test_session,
    )

    chain.to_database("large_table", connection, batch_size=1000)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM large_table"))
        count = result.scalar()

    assert count == large_size

    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text(
                "SELECT * FROM large_table WHERE id IN (0, 999, 5000, 9999) ORDER BY id"
            )
        )
        rows = result.fetchall()

    assert len(rows) == 4
    assert rows[0] == (0, "value_0", 0.0)
    assert rows[1] == (999, "value_999", 2497.5)
    assert rows[2] == (5000, "value_5000", 12500.0)
    assert rows[3] == (9999, "value_9999", 24997.5)


def test_to_database_on_conflict_ignore(db_engine, test_session):
    """Test to_database with on_conflict='ignore' for duplicate handling."""
    with db_engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE conflict_test (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO conflict_test VALUES "
                    "(1, 'Alice', 100), (2, 'Bob', 200), (3, 'Charlie', 300)"
                )
            )

    conflict_chain = dc.read_values(
        id=[2, 3, 4, 5],  # 2 and 3 will conflict
        name=["Bob_Updated", "Charlie_Updated", "Diana", "Eve"],
        value=[999, 888, 400, 500],
        session=test_session,
    )

    conflict_chain.to_database("conflict_test", db_engine, on_conflict="ignore")

    with db_engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("SELECT * FROM conflict_test ORDER BY id")
        )
        rows = result.fetchall()

    assert len(rows) == 5
    assert rows[0] == (1, "Alice", 100)  # Original
    assert rows[1] == (2, "Bob", 200)  # Original (conflict ignored)
    assert rows[2] == (3, "Charlie", 300)  # Original (conflict ignored)
    assert rows[3] == (4, "Diana", 400)  # New
    assert rows[4] == (5, "Eve", 500)  # New


def test_to_database_on_conflict_update(db_engine, test_session):
    """Test to_database with on_conflict='update' for duplicate handling."""
    with db_engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE update_test (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO update_test VALUES "
                    "(1, 'Alice', 100), (2, 'Bob', 200), (3, 'Charlie', 300)"
                )
            )

    update_chain = dc.read_values(
        id=[2, 3, 4, 5],  # 2 and 3 will update existing records
        name=["Bob_Updated", "Charlie_Updated", "Diana", "Eve"],
        value=[999, 888, 400, 500],
        session=test_session,
    )

    update_chain.to_database("update_test", db_engine, on_conflict="update")

    with db_engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT * FROM update_test ORDER BY id"))
        rows = result.fetchall()

    assert len(rows) == 5
    assert rows[0] == (1, "Alice", 100)  # Original (unchanged)
    assert rows[1] == (2, "Bob_Updated", 999)  # Updated
    assert rows[2] == (3, "Charlie_Updated", 888)  # Updated
    assert rows[3] == (4, "Diana", 400)  # New
    assert rows[4] == (5, "Eve", 500)  # New


def test_to_database_table_exists_different_schema(db_engine, test_session):
    """Test to_database when table exists but has different schema."""
    with db_engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE schema_mismatch (
                    id INTEGER,
                    old_column TEXT,
                    extra_column REAL
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO schema_mismatch VALUES (1, 'existing', 1.5)"
                )
            )

    chain = dc.read_values(
        id=[2, 3],
        name=["Alice", "Bob"],  # Different column name
        session=test_session,
    )

    with pytest.raises(sqlalchemy.exc.OperationalError):
        chain.to_database("schema_mismatch", db_engine)


def test_to_database_transaction_rollback_on_error(db_engine, test_session):
    """Test that transaction is rolled back when an exception occurs."""
    with db_engine.connect() as conn:
        with conn.begin():
            conn.execute(
                sqlalchemy.text("""
                CREATE TABLE strict_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL
                )
            """)
            )
            conn.execute(
                sqlalchemy.text(
                    "INSERT INTO strict_table VALUES (1, 'Initial', 'initial@test.com')"
                )
            )

    with db_engine.connect() as conn:
        initial_count = conn.execute(
            sqlalchemy.text("SELECT COUNT(*) FROM strict_table")
        ).scalar()

    assert initial_count == 1

    chain = dc.read_values(
        id=[2, 3, 4],
        name=["Alice", "Bob", "Charlie"],
        # Second email is duplicate and will cause violation
        email=["alice@test.com", "initial@test.com", "charlie@test.com"],
        session=test_session,
    )

    with pytest.raises(sqlalchemy.exc.IntegrityError):
        chain.to_database("strict_table", db_engine)

    # Verify transaction was rolled back - no new records should be inserted
    with db_engine.connect() as conn:
        final_count = conn.execute(
            sqlalchemy.text("SELECT COUNT(*) FROM strict_table")
        ).scalar()

    assert final_count == 1  # Only the initial record should remain


def test_to_database_empty_chain(connection, test_session):
    """Test to_database with an empty DataChain."""
    chain = dc.read_values(
        id=[],
        name=[],
        session=test_session,
    )

    chain.to_database("empty_table", connection)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM empty_table"))
        count = result.scalar()

    assert count == 0


def test_to_database_column_mapping(connection, test_session):
    """Test to_database with column mapping functionality."""
    chain = dc.read_values(
        internal_id=[1, 2, 3],
        full_name=["Alice Smith", "Bob Jones", "Charlie Brown"],
        session=test_session,
    )

    column_mapping = {"internal_id": "id", "full_name": "name"}

    chain.to_database("mapped_table", connection, column_mapping=column_mapping)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("SELECT id, name FROM mapped_table ORDER BY id")
        )
        rows = result.fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice Smith")
    assert rows[1] == (2, "Bob Jones")
    assert rows[2] == (3, "Charlie Brown")


def test_to_database_column_mapping_skip_columns(connection, test_session):
    """Test column mapping with explicit column skipping using None."""
    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        age=[25, 30, 35],
        internal_notes=["note1", "note2", "note3"],
        session=test_session,
    )

    column_mapping = {
        "id": "user_id",
        "name": "full_name",
        "age": None,  # Skip this column
        "internal_notes": None,  # Skip this column too
    }

    chain.to_database(
        "skipped_columns_table", connection, column_mapping=column_mapping
    )

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("PRAGMA table_info(skipped_columns_table)")
        )
        columns = [row[1] for row in result.fetchall()]

        assert "user_id" in columns
        assert "full_name" in columns
        assert "age" not in columns
        assert "internal_notes" not in columns

        result = conn.execute(
            sqlalchemy.text(
                "SELECT user_id, full_name FROM skipped_columns_table ORDER BY user_id"
            )
        )
        rows = result.fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice")
    assert rows[1] == (2, "Bob")
    assert rows[2] == (3, "Charlie")


def test_to_database_column_mapping_defaultdict(connection, test_session):
    """Test column mapping using defaultdict to skip columns by default."""
    from collections import defaultdict

    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        age=[25, 30, 35],
        internal_id=["i1", "i2", "i3"],
        secret_key=["sk1", "sk2", "sk3"],
        session=test_session,
    )

    column_mapping = defaultdict(lambda: None)
    column_mapping.update(
        {
            "id": "user_id",
            "name": "user_name",
        }
    )

    chain.to_database("defaultdict_table", connection, column_mapping=column_mapping)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("PRAGMA table_info(defaultdict_table)"))
        columns = [row[1] for row in result.fetchall()]

        assert "user_id" in columns
        assert "user_name" in columns
        assert "age" not in columns
        assert "internal_id" not in columns
        assert "secret_key" not in columns

        result = conn.execute(
            sqlalchemy.text(
                "SELECT user_id, user_name FROM defaultdict_table ORDER BY user_id"
            )
        )
        rows = result.fetchall()

    assert len(rows) == 3
    assert rows[0] == (1, "Alice")
    assert rows[1] == (2, "Bob")
    assert rows[2] == (3, "Charlie")


def test_to_database_column_mapping_defaultdict_with_datachain_format(
    connection, test_session
):
    """Test defaultdict column mapping with DataChain format (dots) in nested fields."""
    from collections import defaultdict

    class NestedInfo(dc.DataModel):
        value: str
        priority: int

    chain = dc.read_values(
        id=[1, 2],
        name=["Alice", "Bob"],
        age=[25, 30],
        nested_info=[
            NestedInfo(value="important", priority=1),
            NestedInfo(value="normal", priority=2),
        ],
        secret_field=["secret1", "secret2"],
        session=test_session,
    )

    column_mapping = defaultdict(lambda: None)
    column_mapping.update(
        {
            "id": "user_id",
            "name": "user_name",
            "nested_info.value": "info_value",  # DataChain format with dots
            # Skip age, nested_info.priority, and secret_field
        }
    )

    chain.to_database(
        "defaultdict_datachain_table", connection, column_mapping=column_mapping
    )

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("PRAGMA table_info(defaultdict_datachain_table)")
        )
        columns = [row[1] for row in result.fetchall()]

        assert "user_id" in columns
        assert "user_name" in columns
        assert "info_value" in columns
        # These should be skipped
        assert "age" not in columns
        assert "nested_info__priority" not in columns
        assert "secret_field" not in columns

        result = conn.execute(
            sqlalchemy.text(
                "SELECT user_id, user_name, info_value "
                "FROM defaultdict_datachain_table ORDER BY user_id"
            )
        )
        rows = result.fetchall()

    assert len(rows) == 2
    assert rows[0] == (1, "Alice", "important")
    assert rows[1] == (2, "Bob", "normal")


def test_to_database_column_mapping_complex_nested_names(connection, test_session):
    """Test column mapping with complex/nested column names."""

    class NestedData(dc.DataModel):
        value: str
        metadata: dict

    chain = dc.read_values(
        id=[1, 2, 3],
        user_profile_name=["Alice", "Bob", "Charlie"],
        system_config_theme=["dark", "light", "auto"],
        nested_data=[
            NestedData(value="data1", metadata={"type": "A"}),
            NestedData(value="data2", metadata={"type": "B"}),
            NestedData(value="data3", metadata={"type": "C"}),
        ],
        session=test_session,
    )

    column_mapping = {
        "id": "id",
        "user_profile_name": "name",
        "system_config_theme": "theme",
        "nested_data.value": "data_value",  # Use DataChain format with dots
        "nested_data.metadata": "data_meta",  # Use DataChain format with dots
    }

    chain.to_database(
        "complex_mapping_table", connection, column_mapping=column_mapping
    )

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(
            sqlalchemy.text("PRAGMA table_info(complex_mapping_table)")
        )
        columns = [row[1] for row in result.fetchall()]

        assert "id" in columns
        assert "name" in columns
        assert "theme" in columns
        assert "data_value" in columns
        assert "data_meta" in columns
        # Original complex names should not be present
        assert "user_profile_name" not in columns
        assert "system_config_theme" not in columns

        result = conn.execute(
            sqlalchemy.text(
                "SELECT id, name, theme, data_value "
                "FROM complex_mapping_table ORDER BY id"
            )
        )
        rows = result.fetchall()

    assert len(rows) == 3
    assert rows[0][:4] == (1, "Alice", "dark", "data1")
    assert rows[1][:4] == (2, "Bob", "light", "data2")
    assert rows[2][:4] == (3, "Charlie", "auto", "data3")


def test_to_database_column_mapping_datachain_format_backward_compatibility(
    connection, test_session
):
    """Test that both DataChain format (dots) and database format (underscores)
    work in column mapping."""

    class NestedData(dc.DataModel):
        value: str
        config: dict

    chain = dc.read_values(
        id=[1, 2],
        simple_field=["test1", "test2"],
        nested_data=[
            NestedData(value="val1", config={"setting": 1}),
            NestedData(value="val2", config={"setting": 2}),
        ],
        session=test_session,
    )

    # Test DataChain format (dots) - this should work
    column_mapping_dots = {
        "id": "record_id",
        "simple_field": "simple",
        "nested_data.value": "nested_val",
        "nested_data.config": "nested_cfg",
    }

    chain.to_database(
        "dots_format_table", connection, column_mapping=column_mapping_dots
    )

    # Test database format (double underscores) - this should also work
    column_mapping_underscores = {
        "id": "record_id",
        "simple_field": "simple",
        "nested_data__value": "nested_val",
        "nested_data__config": "nested_cfg",
    }

    chain.to_database(
        "underscores_format_table",
        connection,
        column_mapping=column_mapping_underscores,
    )

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result_dots = conn.execute(
            sqlalchemy.text(
                "SELECT record_id, simple, nested_val FROM dots_format_table "
                "ORDER BY record_id"
            )
        )
        rows_dots = result_dots.fetchall()

        result_underscores = conn.execute(
            sqlalchemy.text(
                "SELECT record_id, simple, nested_val FROM underscores_format_table "
                "ORDER BY record_id"
            )
        )
        rows_underscores = result_underscores.fetchall()

        # Both should have identical data
        assert len(rows_dots) == 2
        assert len(rows_underscores) == 2
        assert rows_dots == rows_underscores
        assert rows_dots[0][:3] == (1, "test1", "val1")
        assert rows_dots[1][:3] == (2, "test2", "val2")


def test_to_database_invalid_on_conflict_value(connection, test_session):
    """Test that invalid on_conflict values raise ValueError."""
    chain = dc.read_values(
        id=[1, 2, 3],
        name=["Alice", "Bob", "Charlie"],
        session=test_session,
    )

    with pytest.raises(ValueError, match="on_conflict must be 'ignore' or 'update'"):
        chain.to_database("test_table", connection, on_conflict="invalid")


def test_to_database_with_null_values(connection, test_session):
    """Test to_database handles NULL values correctly."""
    chain = dc.read_values(
        id=[1, 2, 3, 4],
        name=["Alice", None, "Charlie", "Diana"],
        age=[25, 30, None, 28],
        session=test_session,
    )

    chain.to_database("null_table", connection)

    engine = _get_engine_from_connection(connection)
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT * FROM null_table ORDER BY id"))
        rows = result.fetchall()

    assert len(rows) == 4
    assert rows[0] == (1, "Alice", 25)
    assert rows[1] == (2, None, 30)
    assert rows[2] == (3, "Charlie", None)
    assert rows[3] == (4, "Diana", 28)


def test_to_database_table_cleanup_on_map_exception(db_engine, test_session):
    """Test that newly created tables are cleaned up when an exception occurs
    during DataChain iteration using map function."""
    table_name = "cleanup_map_test_table"

    inspector = sqlalchemy.inspect(db_engine)
    assert table_name not in inspector.get_table_names()

    def process_item(id_val):
        # Process first few rows normally, then fail on the third item
        # This ensures table creation happens first, but exception occurs during
        # processing
        if id_val >= 3:
            raise ValueError("Processing failed on third item")
        return f"item_{id_val}"

    chain = dc.read_values(
        id=[1, 2, 3, 4, 5],
        session=test_session,
    ).map(process_item, params=["id"], output={"data": str})

    # The export should fail during processing, after table creation
    with pytest.raises(dc.DataChainError, match="Processing failed on third item"):
        chain.to_database(table_name, db_engine)

    # Verify the table was cleaned up (removed) due to the exception
    inspector = sqlalchemy.inspect(db_engine)
    assert table_name not in inspector.get_table_names(), (
        f"Table {table_name} should have been cleaned up after processing exception"
    )


def _get_engine_from_connection(connection: Any) -> sqlalchemy.Engine:
    """Helper function to get SQLAlchemy engine from various connection types."""
    if isinstance(connection, str):
        return sqlalchemy.create_engine(connection)
    if isinstance(connection, sqlalchemy.Engine):
        return connection
    if isinstance(connection, sqlalchemy.Connection):
        return connection.engine
    if isinstance(connection, Session):
        if connection.bind is None:
            raise ValueError("Session has no bound engine")
        # Session.bind can be Engine or Connection, we need Engine
        if isinstance(connection.bind, sqlalchemy.Engine):
            return connection.bind
        if isinstance(connection.bind, sqlalchemy.Connection):
            return connection.bind.engine
        raise TypeError(f"Unexpected bind type: {type(connection.bind)}")
    if hasattr(connection, "execute"):  # sqlite3.Connection
        return sqlalchemy.create_engine("sqlite://", creator=lambda: connection)
    raise TypeError(f"Unsupported connection type: {type(connection)}")
