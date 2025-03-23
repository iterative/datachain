from typing import TYPE_CHECKING

from datachain.catalog import Catalog

if TYPE_CHECKING:
    from datachain.data_storage import AbstractWarehouse


def test_catalog_warehouse_ready_callback(mocker, warehouse, metastore):
    spy = mocker.spy(warehouse, "is_ready")

    def callback(warehouse: "AbstractWarehouse"):
        assert warehouse.is_ready()

    catalog = Catalog(metastore, warehouse, warehouse_ready_callback=callback)

    spy.assert_not_called()

    _ = catalog.warehouse

    spy.assert_called_once()
    spy.reset_mock()

    _ = catalog.warehouse

    spy.assert_not_called()
