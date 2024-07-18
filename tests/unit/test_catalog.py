from textwrap import dedent
from typing import TYPE_CHECKING

from datachain.catalog import Catalog

if TYPE_CHECKING:
    from datachain.data_storage import AbstractWarehouse


def test_compile_query_script_no_feature_class(catalog):
    script = dedent(
        """
        from datachain.query import C, DatasetQuery, asUDF
        DatasetQuery("s3://bkt/dir1")
        """
    ).strip()
    feature, result = catalog.compile_query_script(script, "tmpfeature")
    expected = dedent(
        """
        from datachain.query import C, DatasetQuery, asUDF
        import datachain.query.dataset
        datachain.query.dataset.query_wrapper(
        DatasetQuery('s3://bkt/dir1'))
        """
    ).strip()
    assert feature is None
    assert result == expected


def test_compile_query_script_with_feature_class(catalog):
    script = dedent(
        """
        from datachain.query import C, DatasetQuery, asUDF
        from datachain.lib.data_model import DataModel as FromAlias
        from datachain.lib.data_model import DataModel
        import datachain.lib.data_model.DataModel as DirectImportedFeature
        import datachain

        class NormalClass:
            t = 1

        class SFClass(FromAlias):
            emb: float

        class DirectImport(DirectImportedFeature):
            emb: float

        class FullImport(datachain.lib.data_model.DataModel):
            emb: float

        class Embedding(DataModel):
            emb: float

        DatasetQuery("s3://bkt/dir1")
        """
    ).strip()
    feature, result = catalog.compile_query_script(script, "tmpfeature")
    expected_feature = dedent(
        """
        from datachain.query import C, DatasetQuery, asUDF
        from datachain.lib.data_model import DataModel as FromAlias
        from datachain.lib.data_model import DataModel
        import datachain.lib.data_model.DataModel as DirectImportedFeature
        import datachain
        import datachain.query.dataset

        class SFClass(FromAlias):
            emb: float

        class DirectImport(DirectImportedFeature):
            emb: float

        class FullImport(datachain.lib.data_model.DataModel):
            emb: float

        class Embedding(DataModel):
            emb: float
        """
    ).strip()
    expected_result = dedent(
        """
        from datachain.query import C, DatasetQuery, asUDF
        from datachain.lib.data_model import DataModel as FromAlias
        from datachain.lib.data_model import DataModel
        import datachain.lib.data_model.DataModel as DirectImportedFeature
        import datachain
        import datachain.query.dataset
        from tmpfeature import *

        class NormalClass:
            t = 1
        datachain.query.dataset.query_wrapper(
        DatasetQuery('s3://bkt/dir1'))
        """
    ).strip()

    assert feature == expected_feature
    assert result == expected_result


def test_compile_query_script_with_decorator(catalog):
    script = dedent(
        """
        import os
        from datachain.query import C, DatasetQuery, udf
        from datachain.sql.types import Float, Float32, Int, String, Binary

        @udf(
            params=("name", ),
            output={"num": Float, "bin": Binary}
        )
        def my_func1(name):
            x = 3.14
            int_example = 25
            bin = int_example.to_bytes(2, "big")
            return (x, bin)

        print("Test ENV = ", os.environ['TEST_ENV'])
        ds = DatasetQuery("s3://dql-small/*.jpg") \
                .add_signals(my_func1)

        ds
        """
    ).strip()
    feature, result = catalog.compile_query_script(script, "tmpfeature")

    expected_result = dedent(
        """
        import os
        from datachain.query import C, DatasetQuery, udf
        from datachain.sql.types import Float, Float32, Int, String, Binary
        import datachain.query.dataset

        @udf(params=('name',), output={'num': Float, 'bin': Binary})
        def my_func1(name):
            x = 3.14
            int_example = 25
            bin = int_example.to_bytes(2, 'big')
            return (x, bin)
        print('Test ENV = ', os.environ['TEST_ENV'])
        ds = DatasetQuery('s3://dql-small/*.jpg').add_signals(my_func1)
        datachain.query.dataset.query_wrapper(
        ds)
        """
    ).strip()

    assert feature is None
    assert result == expected_result


def test_catalog_warehouse_ready_callback(mocker, warehouse, id_generator, metastore):
    spy = mocker.spy(warehouse, "is_ready")

    def callback(warehouse: "AbstractWarehouse"):
        assert warehouse.is_ready()

    catalog = Catalog(
        id_generator, metastore, warehouse, warehouse_ready_callback=callback
    )

    spy.assert_not_called()

    _ = catalog.warehouse

    spy.assert_called_once()
    spy.reset_mock()

    _ = catalog.warehouse

    spy.assert_not_called()
