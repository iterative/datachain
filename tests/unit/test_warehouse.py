import base64
import pickle

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteWarehouse,
)


def test_serialize(sqlite_db):
    obj = SQLiteWarehouse(sqlite_db)
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteWarehouse)
    assert obj2.db.db_file == sqlite_db.db_file
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization
    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(SQLiteWarehouse.init_after_clone)
    assert args == []
    assert str(kwargs["db_clone_params"]) == str(sqlite_db.clone_params())

    # Test deserialization
    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteWarehouse)
    assert obj3.db.db_file == sqlite_db.db_file
    assert obj3.clone_params() == obj.clone_params()


def test_is_temp_table_name(warehouse):
    assert warehouse.is_temp_table_name("tmp_vc12F") is True
    assert warehouse.is_temp_table_name("udf_jh653") is True
    assert warehouse.is_temp_table_name("ds_my_dataset") is False
    assert warehouse.is_temp_table_name("src_my_bucket") is False
    assert warehouse.is_temp_table_name("ds_ds_my_query_script_1_1") is False
