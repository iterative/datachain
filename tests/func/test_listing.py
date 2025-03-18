import pytest

from datachain.lib.dc import DataChain
from datachain.lib.listing import list_bucket, parse_listing_uri
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


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs", "file"],
    indirect=True,
)
def test_parse_listing_uri(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(
        f"{ctc.src_uri}/dogs", catalog.client_config
    )
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs/"
    if cloud_type == "file":
        assert listing_path == ""
    else:
        assert listing_path == "dogs/"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_parse_listing_uri_with_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(
        f"{ctc.src_uri}/dogs/*", catalog.client_config
    )
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs"
    assert listing_path == "dogs/*"
