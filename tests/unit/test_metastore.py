import base64
import pickle

import pytest

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SCHEMA_VERSION, SQLiteMetastore
from datachain.dataset import StorageURI
from datachain.error import OutdatedDatabaseSchemaError
from tests.conftest import cleanup_sqlite_db


def test_sqlite_metastore(sqlite_db):
    uri = StorageURI("s3://bucket")

    obj = SQLiteMetastore(uri, sqlite_db)
    assert obj.uri == uri
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteMetastore)
    assert obj2.uri == uri
    assert obj2.db.db_file == sqlite_db.db_file
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization
    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(SQLiteMetastore.init_after_clone)
    assert args == []
    assert kwargs["uri"] == uri
    assert str(kwargs["db_clone_params"]) == str(sqlite_db.clone_params())

    # Test deserialization
    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteMetastore)
    assert obj3.uri == uri
    assert obj3.db.db_file == sqlite_db.db_file
    assert obj3.clone_params() == obj.clone_params()


def test_outdated_schema_meta_not_present():
    metastore = SQLiteMetastore(db_file=":memory:")

    metastore.db.drop_table(metastore._meta)

    with pytest.raises(OutdatedDatabaseSchemaError):
        metastore = SQLiteMetastore(db_file=":memory:")

    cleanup_sqlite_db(metastore.db.clone(), metastore.default_table_names)
    metastore.close_on_exit()


def test_outdated_schema():
    metastore = SQLiteMetastore(db_file=":memory:")

    # update schema version to be lower than current one
    stmt = (
        metastore._meta.update()
        .where(metastore._meta.c.id == 1)
        .values(schema_version=SCHEMA_VERSION - 1)
    )
    metastore.db.execute(stmt)

    with pytest.raises(OutdatedDatabaseSchemaError):
        metastore = SQLiteMetastore(db_file=":memory:")

    cleanup_sqlite_db(metastore.db.clone(), metastore.default_table_names)
    metastore.close_on_exit()
