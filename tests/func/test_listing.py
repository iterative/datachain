from datachain.client.local import FileClient
from datachain.lib.dc import DataChain
from datachain.lib.listing import list_bucket
from tests.data import ENTRIES


def test_listing_generator(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog

    uri = f"{ctc.src_uri}/cats"

    dc = DataChain.from_records(DataChain.DEFAULT_FILE_RECORD).gen(
        file=list_bucket(uri, client_config=ctc.catalog.client_config)
    )
    assert dc.count() == 2

    entries = sorted(
        [e for e in ENTRIES if e.path.startswith("cats/")], key=lambda e: e.path
    )
    files = sorted(dc.collect("file"), key=lambda f: f.path)

    for cat_file, cat_entry in zip(files, entries):
        if cloud_type == "file":
            root_uri = FileClient.root_path().as_uri()
            assert cat_file.source == root_uri
            assert cat_file.path == f"{ctc.src_uri}/{cat_entry.path}".removeprefix(
                root_uri
            )
        else:
            assert cat_file.source == cloud_test_catalog.src_uri
            assert cat_file.path == cat_entry.path
        assert cat_file.size == cat_entry.size
        assert cat_file.is_latest == cat_entry.is_latest
        assert cat_file.location is None
