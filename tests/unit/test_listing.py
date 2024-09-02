import posixpath
from datetime import datetime, timedelta, timezone

import pytest

from datachain.catalog import Catalog
from datachain.catalog.catalog import DataSource
from datachain.lib.listing import (
    LISTING_TTL,
    is_listing_dataset,
    is_listing_expired,
    is_listing_subset,
    listing_uri_from_name,
    parse_listing_uri,
)
from datachain.node import DirType, Entry, get_path
from tests.utils import skip_if_not_sqlite

TREE = {
    "dir1": {
        "d2": {None: ["file1.csv", "file2.csv"]},
        None: ["dataset.csv"],
    },
    "dir2": {None: ["diagram.png"]},
    None: ["users.csv"],
}


def _tree_to_entries(tree: dict, path=""):
    for k, v in tree.items():
        if k:
            dir_path = posixpath.join(path, k)
            yield from _tree_to_entries(v, dir_path)
        else:
            for fname in v:
                yield Entry.from_file(get_path(path, fname))


@pytest.fixture
def listing(id_generator, metastore, warehouse):
    catalog = Catalog(
        id_generator=id_generator, metastore=metastore, warehouse=warehouse
    )
    lst, _ = catalog.enlist_source("s3://whatever", 1234, skip_indexing=True)
    lst.insert_entries(_tree_to_entries(TREE))
    lst.insert_entries_done()
    return lst


def test_resolve_path_in_root(listing):
    node = listing.resolve_path("dir1")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "dir1"
    assert node.size == 0


def test_path_resolving_nested(listing):
    node = listing.resolve_path("dir1/d2/file2.csv")
    assert node.dir_type == DirType.FILE
    assert node.name == "file2.csv"
    assert not node.is_dir

    node = listing.resolve_path("dir1/d2")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "d2"


def test_resolve_not_existing_path(listing):
    with pytest.raises(FileNotFoundError):
        listing.resolve_path("dir1/fake-file-name")


def test_resolve_root(listing):
    node = listing.resolve_path("")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == ""
    assert node.size == 0


def test_dir_ends_with_slash(listing):
    node = listing.resolve_path("dir1/")
    assert node.dir_type == DirType.DIR
    assert node.is_dir
    assert node.name == "dir1"


def test_file_ends_with_slash(listing):
    with pytest.raises(FileNotFoundError):
        listing.resolve_path("dir1/dataset.csv/")


def _match_filenames(nodes, expected_names):
    assert len(nodes) == len(expected_names)
    names = (node.name for node in nodes)
    assert set(names) == set(expected_names)


def test_basic_expansion(listing):
    nodes = listing.expand_path("*")
    _match_filenames(nodes, ["dir1", "dir2", "users.csv"])


def test_subname_expansion(listing):
    nodes = listing.expand_path("di*/")
    _match_filenames(nodes, ["dir1", "dir2"])


@skip_if_not_sqlite
def test_multilevel_expansion(listing):
    nodes = listing.expand_path("dir[1,2]/d*")
    _match_filenames(nodes, ["dataset.csv", "diagram.png", "d2"])


def test_expand_root(listing):
    nodes = listing.expand_path("")
    assert len(nodes) == 1
    assert nodes[0].dir_type == DirType.DIR
    assert nodes[0].is_dir


def test_list_dir(listing):
    dir1 = listing.resolve_path("dir1/")
    names = listing.ls_path(dir1, ["path"])
    assert {n[0] for n in names} == {"dir1/d2", "dir1/dataset.csv"}


def test_list_file(listing):
    file = listing.resolve_path("dir1/dataset.csv")
    src = DataSource(listing, file)
    results = list(src.ls(["sys__id", "name", "dir_type"]))
    assert {r[1] for r in results} == {"dataset.csv"}
    assert results[0][0] == file.sys__id
    assert results[0][1] == file.name
    assert results[0][2] == DirType.FILE


def test_subtree(listing):
    dir1 = listing.resolve_path("dir1/")
    nodes = listing.subtree_files(dir1)
    subtree_files = ["dataset.csv", "file1.csv", "file2.csv"]
    _match_filenames([nwp.n for nwp in nodes], subtree_files)


def test_subdirs(listing):
    dirs = list(listing.get_dirs_by_parent_path(""))
    _match_filenames(dirs, ["dir1", "dir2"])


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_parse_listing_uri(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(
        f"{ctc.src_uri}/dogs", catalog.cache, catalog.client_config
    )
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs"
    assert listing_path == "dogs"


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_parse_listing_uri_with_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    dataset_name, listing_uri, listing_path = parse_listing_uri(
        f"{ctc.src_uri}/dogs/*", catalog.cache, catalog.client_config
    )
    assert dataset_name == f"lst__{ctc.src_uri}/dogs/"
    assert listing_uri == f"{ctc.src_uri}/dogs"
    assert listing_path == "dogs/*"


@pytest.mark.parametrize(
    "name,is_listing",
    [
        ("lst__s3://my-bucket", True),
        ("lst__file:///my-folder/dir1", True),
        ("s3://my-bucket", False),
        ("my-dataset", False),
    ],
)
def test_is_listing_dataset(name, is_listing):
    assert is_listing_dataset(name) is is_listing


def test_listing_uri_from_name():
    assert listing_uri_from_name("lst__s3://my-bucket") == "s3://my-bucket"
    with pytest.raises(ValueError):
        listing_uri_from_name("s3://my-bucket")


@pytest.mark.parametrize(
    "date,is_expired",
    [
        (datetime.now(timezone.utc), False),
        (datetime.now(timezone.utc) - timedelta(seconds=LISTING_TTL + 1), True),
    ],
)
def test_is_listing_expired(date, is_expired):
    assert is_listing_expired(date) is is_expired


@pytest.mark.parametrize(
    "ds1_name,ds2_name,is_subset",
    [
        ("lst__s3://my-bucket/animals/", "lst__s3://my-bucket/animals/dogs/", True),
        ("lst__s3://my-bucket/animals/", "lst__s3://my-bucket/animals/", True),
        ("lst__s3://my-bucket/", "lst__s3://my-bucket/", True),
        ("lst__s3://my-bucket/cats/", "lst__s3://my-bucket/animals/dogs/", False),
        ("lst__s3://my-bucket/dogs/", "lst__s3://my-bucket/animals/", False),
        ("lst__s3://my-bucket/animals/", "lst__s3://other-bucket/animals/", False),
    ],
)
def test_listing_subset(ds1_name, ds2_name, is_subset):
    assert is_listing_subset(ds1_name, ds2_name) is is_subset
