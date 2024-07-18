import os

import pytest

from datachain.lib.dc import DataChain
from datachain.lib.file import File


@pytest.mark.parametrize("anon", [True, False])
def test_catalog_anon(catalog, anon):
    chain = (
        DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/", anon=anon)
        .limit(5)
        .save("test_catalog_anon")
    )
    assert chain.catalog.client_config.get("anon", False) is anon


def test_from_storage(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
    assert dc.count() == 7


@pytest.mark.parametrize("use_cache", [False, True])
def test_map_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    dc = (
        DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
        .settings(cache=use_cache)
        .map(signal=new_signal)
    )
    expected = {
        "description -> Cats and Dogs",
        "cat1 -> meow",
        "cat2 -> mrow",
        "dog1 -> woof",
        "dog2 -> arf",
        "dog3 -> bark",
        "dog4 -> ruff",
    }
    assert set(dc.iterate_one("signal")) == expected
    for file in dc.iterate_one("file"):
        assert bool(file.get_local_path()) is use_cache


@pytest.mark.parametrize("use_cache", [False, True])
def test_read_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    dc = DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
    for file in dc.settings(cache=use_cache).iterate_one("file"):
        assert file.get_local_path() is None
        file.read()
        assert bool(file.get_local_path()) is use_cache


@pytest.mark.parametrize("placement", ["fullpath", "filename"])
@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_export_files(tmp_dir, cloud_test_catalog, placement, use_map, use_cache):
    ctc = cloud_test_catalog
    df = DataChain.from_storage(ctc.src_uri)
    if use_map:
        df.export_files(tmp_dir / "output", placement=placement, use_cache=use_cache)
        df.map(
            res=lambda file: file.export(
                tmp_dir / "output", placement=placement, use_cache=use_cache
            )
        ).exec()
    else:
        df.export_files(tmp_dir / "output", placement=placement)

    expected = {
        "description": "Cats and Dogs",
        "cat1": "meow",
        "cat2": "mrow",
        "dog1": "woof",
        "dog2": "arf",
        "dog3": "bark",
        "dog4": "ruff",
    }

    for file in df.collect_one("file"):
        if placement == "filename":
            file_path = file.name
        else:
            file_path = file.get_full_name()
        with open(tmp_dir / "output" / file_path) as f:
            assert f.read() == expected[file.name]


def test_export_files_filename_placement_not_unique_files(tmp_dir, catalog):
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"
    bucket_name = "mybucket"
    files = ["dir1/a.json", "dir1/dir2/a.json"]

    # create bucket dir with duplicate file names
    bucket_dir = tmp_dir / bucket_name
    bucket_dir.mkdir(parents=True)
    for file_path in files:
        file_path = bucket_dir / file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as fd:
            fd.write(data)

    df = DataChain.from_storage((tmp_dir / bucket_name).as_uri())
    with pytest.raises(ValueError):
        df.export_files(tmp_dir / "output", placement="filename")
