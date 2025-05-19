def test_dataset_stats_no_table(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.warehouse.drop_dataset_rows_table(dogs_dataset, version="1.0.0")
    num_objects, size = catalog.warehouse.dataset_stats(dogs_dataset, version="1.0.0")
    assert num_objects is None
    assert size is None
