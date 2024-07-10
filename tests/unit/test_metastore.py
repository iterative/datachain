import base64
import pickle

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    SQLiteIDGenerator,
    SQLiteMetastore,
)
from datachain.storage import StorageURI


def test_sqlite_metastore():
    db = SQLiteDatabaseEngine.from_db_file(":memory:")
    id_generator = SQLiteIDGenerator(db, table_prefix="prefix")
    uri = StorageURI("s3://bucket")

    obj = SQLiteMetastore(id_generator, uri, 1, db)
    assert obj.id_generator == id_generator
    assert obj.uri == uri
    assert obj.partial_id == 1
    assert obj.db == db

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteMetastore)
    assert obj2.id_generator.db.db_file == obj.id_generator.db.db_file
    assert obj2.id_generator._table_prefix == obj.id_generator._table_prefix
    assert obj2.uri == uri
    assert obj2.partial_id == 1
    assert obj2.db.db_file == db.db_file
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization
    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(SQLiteMetastore.init_after_clone)
    assert args == []
    assert str(kwargs["id_generator_clone_params"]) == str(id_generator.clone_params())
    assert kwargs["uri"] == uri
    assert kwargs["partial_id"] == 1
    assert str(kwargs["db_clone_params"]) == str(db.clone_params())

    # Test deserialization
    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteMetastore)
    assert obj3.id_generator.db.db_file == id_generator.db.db_file
    assert obj3.id_generator._table_prefix == id_generator._table_prefix
    assert obj3.uri == uri
    assert obj3.partial_id == 1
    assert obj3.db.db_file == db.db_file
    assert obj3.clone_params() == obj.clone_params()
