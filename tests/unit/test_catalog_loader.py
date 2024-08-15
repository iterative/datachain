import os
from unittest.mock import patch

import pytest

from datachain.catalog.loader import (
    get_catalog,
    get_distributed_class,
    get_id_generator,
    get_metastore,
    get_warehouse,
)
from datachain.data_storage.sqlite import (
    SQLiteIDGenerator,
    SQLiteMetastore,
    SQLiteWarehouse,
)
from datachain.storage import StorageURI


class DistributedClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_id_generator(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db, table_prefix="prefix")
    assert id_generator.db == sqlite_db
    assert id_generator._table_prefix == "prefix"

    with patch.dict(os.environ, {"DATACHAIN__ID_GENERATOR": id_generator.serialize()}):
        id_generator2 = get_id_generator()
        assert id_generator2
        assert isinstance(id_generator2, SQLiteIDGenerator)
        assert id_generator2._db.db_file == sqlite_db.db_file
        assert id_generator2._table_prefix == "prefix"
        assert id_generator2.clone_params() == id_generator.clone_params()

    with patch.dict(os.environ, {"DATACHAIN__ID_GENERATOR": sqlite_db.serialize()}):
        with pytest.raises(RuntimeError, match="instance of AbstractIDGenerator"):
            get_id_generator()


def test_get_id_generator_in_memory():
    if os.environ.get("DATACHAIN_ID_GENERATOR"):
        with pytest.raises(RuntimeError):
            id_generator = get_id_generator(in_memory=True)
    else:
        id_generator = get_id_generator(in_memory=True)
        assert isinstance(id_generator, SQLiteIDGenerator)
        assert id_generator.db.db_file == ":memory:"
        id_generator.close()


def test_get_metastore(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db, table_prefix="prefix")
    uri = StorageURI("s3://bucket")
    partial_id = 37

    metastore = SQLiteMetastore(id_generator, uri, partial_id, sqlite_db)
    assert metastore.id_generator == id_generator
    assert metastore.uri == uri
    assert metastore.partial_id == partial_id
    assert metastore.db == sqlite_db

    with patch.dict(os.environ, {"DATACHAIN__METASTORE": metastore.serialize()}):
        metastore2 = get_metastore(None)
        assert metastore2
        assert isinstance(metastore2, SQLiteMetastore)
        assert metastore2.id_generator._db.db_file == metastore.id_generator._db.db_file
        assert (
            metastore2.id_generator._table_prefix
            == metastore.id_generator._table_prefix
        )
        assert metastore2.uri == uri
        assert metastore2.partial_id == partial_id
        assert metastore2.db.db_file == sqlite_db.db_file
        assert metastore2.clone_params() == metastore.clone_params()

    with patch.dict(os.environ, {"DATACHAIN__METASTORE": sqlite_db.serialize()}):
        with pytest.raises(RuntimeError, match="instance of AbstractMetastore"):
            get_metastore(None)


def test_get_metastore_in_memory():
    if os.environ.get("DATACHAIN_METASTORE"):
        id_generator = get_id_generator()
        with pytest.raises(RuntimeError):
            metastore = get_metastore(id_generator, in_memory=True)
        id_generator.close()
    else:
        id_generator = get_id_generator(in_memory=True)
        metastore = get_metastore(id_generator, in_memory=True)
        assert isinstance(metastore, SQLiteMetastore)
        assert metastore.db.db_file == ":memory:"
        metastore.close()
        id_generator.close()


def test_get_warehouse(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db, table_prefix="prefix")

    warehouse = SQLiteWarehouse(id_generator, sqlite_db)
    assert warehouse.id_generator == id_generator
    assert warehouse.db == sqlite_db

    with patch.dict(os.environ, {"DATACHAIN__WAREHOUSE": warehouse.serialize()}):
        warehouse2 = get_warehouse(None)
        assert warehouse2
        assert isinstance(warehouse2, SQLiteWarehouse)
        assert warehouse2.id_generator._db.db_file == warehouse.id_generator._db.db_file
        assert (
            warehouse2.id_generator._table_prefix
            == warehouse.id_generator._table_prefix
        )
        assert warehouse2.db.db_file == sqlite_db.db_file
        assert warehouse2.clone_params() == warehouse.clone_params()

    with patch.dict(os.environ, {"DATACHAIN__WAREHOUSE": sqlite_db.serialize()}):
        with pytest.raises(RuntimeError, match="instance of AbstractWarehouse"):
            get_warehouse(None)


def test_get_warehouse_in_memory():
    if os.environ.get("DATACHAIN_WAREHOUSE"):
        id_generator = get_id_generator()
        with pytest.raises(RuntimeError):
            warehouse = get_warehouse(id_generator, in_memory=True)
        id_generator.close()
    else:
        id_generator = get_id_generator(in_memory=True)
        warehouse = get_warehouse(id_generator, in_memory=True)
        assert isinstance(warehouse, SQLiteWarehouse)
        assert warehouse.db.db_file == ":memory:"
        warehouse.close()
        id_generator.close()


def test_get_distributed_class():
    distributed_args = {"foo": "bar", "baz": "37", "empty": ""}
    env = {
        "DATACHAIN_DISTRIBUTED": "tests.unit.test_catalog_loader.DistributedClass",
        "DATACHAIN_DISTRIBUTED_ARG_FOO": "bar",
        "DATACHAIN_DISTRIBUTED_ARG_BAZ": "37",
        "DATACHAIN_DISTRIBUTED_ARG_EMPTY": "",
    }

    with patch.dict(os.environ, env):
        distributed = get_distributed_class()
        assert distributed
        assert isinstance(distributed, DistributedClass)
        assert distributed.kwargs == distributed_args

    with patch.dict(os.environ, {"DATACHAIN_DISTRIBUTED": ""}):
        with pytest.raises(
            RuntimeError, match="DATACHAIN_DISTRIBUTED import path is required"
        ):
            get_distributed_class()

    with patch.dict(
        os.environ,
        {"DATACHAIN_DISTRIBUTED": "tests.unit.test_catalog_loader.NonExistent"},
    ):
        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            get_distributed_class()

    with patch.dict(os.environ, {"DATACHAIN_DISTRIBUTED": "DistributionClass"}):
        with pytest.raises(
            RuntimeError, match="Invalid DATACHAIN_DISTRIBUTED import path"
        ):
            get_distributed_class()


def test_get_catalog(sqlite_db):
    id_generator = SQLiteIDGenerator(sqlite_db, table_prefix="prefix")
    uri = StorageURI("s3://bucket")
    partial_id = 73
    metastore = SQLiteMetastore(id_generator, uri, partial_id, sqlite_db)
    warehouse = SQLiteWarehouse(id_generator, sqlite_db)
    env = {
        "DATACHAIN__ID_GENERATOR": id_generator.serialize(),
        "DATACHAIN__METASTORE": metastore.serialize(),
        "DATACHAIN__WAREHOUSE": warehouse.serialize(),
    }

    with patch.dict(os.environ, env):
        catalog = get_catalog()
        assert catalog

        assert catalog.id_generator
        assert isinstance(catalog.id_generator, SQLiteIDGenerator)
        assert catalog.id_generator._db.db_file == sqlite_db.db_file
        assert catalog.id_generator._table_prefix == "prefix"
        assert catalog.id_generator.clone_params() == id_generator.clone_params()

        assert catalog.metastore
        assert isinstance(catalog.metastore, SQLiteMetastore)
        assert (
            catalog.metastore.id_generator._db.db_file
            == metastore.id_generator._db.db_file
        )
        assert (
            catalog.metastore.id_generator._table_prefix
            == metastore.id_generator._table_prefix
        )
        assert catalog.metastore.uri == uri
        assert catalog.metastore.partial_id == partial_id
        assert catalog.metastore.db.db_file == sqlite_db.db_file
        assert catalog.metastore.clone_params() == metastore.clone_params()

        assert catalog.warehouse
        assert isinstance(catalog.warehouse, SQLiteWarehouse)
        assert (
            catalog.warehouse.id_generator._db.db_file
            == warehouse.id_generator._db.db_file
        )
        assert (
            catalog.warehouse.id_generator._table_prefix
            == warehouse.id_generator._table_prefix
        )
        assert catalog.warehouse.db.db_file == sqlite_db.db_file
        assert catalog.warehouse.clone_params() == warehouse.clone_params()
