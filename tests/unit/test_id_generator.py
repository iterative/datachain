import base64
import pickle

from sqlalchemy import select

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteDatabaseEngine, SQLiteIDGenerator


def get_rows(id_generator):
    uris = id_generator.db.execute(
        select(id_generator._table.c.uri, id_generator._table.c.last_id)
    ).fetchall()
    return set(uris)


def test_init(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db)
    assert id_generator.db == sqlite_db
    assert id_generator._table_prefix is None
    assert sqlite_db.has_table("id_generator")
    assert get_rows(id_generator) == set()

    id_generator.cleanup_for_tests()
    assert not sqlite_db.has_table("id_generator")


def test_init_empty(tmp_dir):
    id_generator = SQLiteIDGenerator()
    assert id_generator._table_prefix is None
    assert id_generator.db
    assert id_generator.db.has_table("id_generator")
    assert get_rows(id_generator) == set()

    id_generator.cleanup_for_tests()
    assert not id_generator.db.has_table("id_generator")


def test_init_with_prefix(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db, table_prefix="foo")
    assert id_generator.db == sqlite_db
    assert id_generator._table_prefix == "foo"
    assert sqlite_db.has_table("foo_id_generator")
    assert not sqlite_db.has_table("id_generator")
    assert get_rows(id_generator) == set()

    id_generator.cleanup_for_tests()
    assert not sqlite_db.has_table("foo_id_generator")


def test_clone(id_generator):
    clone = id_generator.clone()
    assert clone._table_prefix == id_generator._table_prefix
    assert get_rows(clone) == get_rows(id_generator)

    id_generator.init_id("foo")
    clone.get_next_id("bar")
    assert get_rows(id_generator) == {("foo", 0), ("bar", 1)}
    assert get_rows(clone) == get_rows(id_generator)

    id_generator.cleanup_for_tests()
    assert not id_generator.db.has_table("id_generator")
    assert not clone.db.has_table("id_generator")


def test_clone_params(id_generator):
    func, args, kwargs = id_generator.clone_params()
    clone = func(*args, **kwargs)
    assert clone._table_prefix == id_generator._table_prefix
    assert get_rows(clone) == get_rows(id_generator)

    id_generator.init_id("foo")
    clone.get_next_id("bar")
    assert get_rows(id_generator) == {("foo", 0), ("bar", 1)}
    assert get_rows(clone) == get_rows(id_generator)

    clone.cleanup_for_tests()
    assert not id_generator.db.has_table("id_generator")
    assert not clone.db.has_table("id_generator")


def test_serialize():
    db = SQLiteDatabaseEngine.from_db_file(":memory:")

    obj = SQLiteIDGenerator(db, table_prefix="prefix")
    assert obj.db == db
    assert obj._table_prefix == "prefix"

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteIDGenerator)
    assert obj2.db.db_file == obj.db.db_file
    assert obj2._table_prefix == "prefix"
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization
    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(SQLiteIDGenerator.init_after_clone)
    assert args == []
    assert str(kwargs["db_clone_params"]) == str(db.clone_params())
    assert kwargs["table_prefix"] == "prefix"

    # Test deserialization
    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteIDGenerator)
    assert obj3.db.db_file == db.db_file
    assert obj3._table_prefix == "prefix"


def test_init_id(id_generator):
    assert get_rows(id_generator) == set()

    id_generator.init_id("foo")
    assert get_rows(id_generator) == {("foo", 0)}

    assert id_generator.get_next_id("foo") == 1
    assert get_rows(id_generator) == {("foo", 1)}

    id_generator.init_id("foo")  # second call should not raise an exception
    assert get_rows(id_generator) == {("foo", 1)}

    assert id_generator.get_next_id("foo") == 2
    assert get_rows(id_generator) == {("foo", 2)}

    id_generator.init_id("bar")
    assert get_rows(id_generator) == {("foo", 2), ("bar", 0)}


def test_get_next_id(id_generator):
    assert get_rows(id_generator) == set()

    assert id_generator.get_next_id("foo") == 1
    assert get_rows(id_generator) == {("foo", 1)}

    assert id_generator.get_next_id("foo") == 2
    assert get_rows(id_generator) == {("foo", 2)}

    assert id_generator.get_next_id("bar") == 1
    assert get_rows(id_generator) == {("foo", 2), ("bar", 1)}


def test_get_next_ids(id_generator):
    assert get_rows(id_generator) == set()

    assert id_generator.get_next_ids("foo", 3) == range(1, 4)
    assert get_rows(id_generator) == {("foo", 3)}

    assert id_generator.get_next_ids("foo", 20) == range(4, 24)
    assert get_rows(id_generator) == {("foo", 23)}

    assert id_generator.get_next_ids("bar", 1000) == range(1, 1001)
    assert get_rows(id_generator) == {("foo", 23), ("bar", 1000)}


def test_delete_uri(id_generator):
    assert get_rows(id_generator) == set()

    assert id_generator.get_next_ids("foo", 3) == range(1, 4)
    assert get_rows(id_generator) == {("foo", 3)}

    assert id_generator.get_next_ids("bar", 1000) == range(1, 1001)
    assert get_rows(id_generator) == {("foo", 3), ("bar", 1000)}

    id_generator.delete_uri("foo")
    assert get_rows(id_generator) == {("bar", 1000)}

    id_generator.delete_uri("bar")
    assert get_rows(id_generator) == set()


def test_delete_uris(id_generator):
    assert get_rows(id_generator) == set()

    assert id_generator.get_next_ids("foo", 3) == range(1, 4)
    assert get_rows(id_generator) == {("foo", 3)}

    assert id_generator.get_next_ids("bar", 1000) == range(1, 1001)
    assert get_rows(id_generator) == {("foo", 3), ("bar", 1000)}

    id_generator.delete_uris(["foo", "bar"])
    assert get_rows(id_generator) == set()
