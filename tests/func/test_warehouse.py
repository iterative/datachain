def test_dataset_stats_no_table(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    catalog.warehouse.drop_dataset_rows_table(dogs_dataset, 1)
    num_objects, size = catalog.warehouse.dataset_stats(dogs_dataset, 1)
    assert num_objects is None
    assert size is None
