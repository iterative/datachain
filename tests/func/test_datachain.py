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
