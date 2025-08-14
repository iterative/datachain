from types import GeneratorType

import sqlalchemy as sa

import datachain as dc


def test_dataset_stats_no_table(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.warehouse.drop_dataset_rows_table(dogs_dataset, version="1.0.0")
    num_objects, size = catalog.warehouse.dataset_stats(dogs_dataset, version="1.0.0")
    assert num_objects is None
    assert size is None


def test_dataset_select_paginated_dataset_larger_than_batch_size(test_session):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    chain = dc.read_values(value=list(range(10_000)), session=test_session).save(
        "large"
    )
    table = warehouse.get_table(
        warehouse.dataset_table_name(chain.dataset, chain.dataset.latest_version)
    )
    db_values = chain.to_values("value")

    rows = warehouse.dataset_select_paginated(
        sa.select(table.c.value).order_by(table.c.value), page_size=1000
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 10_000
    (values,) = zip(*rows)
    assert set(values) == set(db_values)
