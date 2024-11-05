from datachain.lib.dc import DataChain
from datachain.lib.listing import list_bucket
from tests.data import ENTRIES


def test_listing_generator(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    catalog = ctc.catalog

    uri = f"{ctc.src_uri}/cats"

    dc = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD).gen(
        file=list_bucket(uri, catalog.cache, client_config=catalog.client_config)
    )
    assert dc.count() == 2

    entries = sorted(
        [e for e in ENTRIES if e.path.startswith("cats/")], key=lambda e: e.path
    )
    files = dc.order_by("file.path").collect("file")

    for cat_file, cat_entry in zip(files, entries):
        assert cat_file.source == ctc.src_uri
        assert cat_file.path == cat_entry.path
        assert cat_file.size == cat_entry.size
        assert cat_file.is_latest == cat_entry.is_latest
        assert cat_file.location is None
