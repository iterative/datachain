import base64
import pickle

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import (
    SQLiteDatabaseEngine,
    SQLiteIDGenerator,
    SQLiteWarehouse,
)


def test_serialize():
    db = SQLiteDatabaseEngine.from_db_file(":memory:")
    id_generator = SQLiteIDGenerator(db, table_prefix="prefix")

    obj = SQLiteWarehouse(id_generator, db)
    assert obj.id_generator == id_generator
    assert obj.db == db

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteWarehouse)
    assert obj2.id_generator.db.db_file == obj.id_generator.db.db_file
    assert obj2.id_generator._table_prefix == obj.id_generator._table_prefix
    assert obj2.db.db_file == db.db_file
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization
    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(SQLiteWarehouse.init_after_clone)
    assert args == []
    assert str(kwargs["id_generator_clone_params"]) == str(id_generator.clone_params())
    assert str(kwargs["db_clone_params"]) == str(db.clone_params())

    # Test deserialization
    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteWarehouse)
    assert obj3.id_generator.db.db_file == id_generator.db.db_file
    assert obj3.id_generator._table_prefix == id_generator._table_prefix
    assert obj3.db.db_file == db.db_file
    assert obj3.clone_params() == obj.clone_params()


def test_is_temp_table_name(warehouse):
    assert warehouse.is_temp_table_name("tmp_vc12F") is True
    assert warehouse.is_temp_table_name("udf_jh653") is True
    assert warehouse.is_temp_table_name("ds_shadow_12345") is True
    assert warehouse.is_temp_table_name("old_ds_shadow") is True
    assert warehouse.is_temp_table_name("ds_my_dataset") is False
    assert warehouse.is_temp_table_name("src_my_bucket") is False
    assert warehouse.is_temp_table_name("ds_ds_my_query_script_1_1") is False


def test_dataset_stats_no_table(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.warehouse.drop_dataset_rows_table(dogs_dataset, 1)
    num_objects, size = catalog.warehouse.dataset_stats(dogs_dataset, 1)
    assert num_objects is None
    assert size is None
