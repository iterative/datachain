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


def test_map_file(cloud_test_catalog):
    ctc = cloud_test_catalog

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    dc = DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog).map(signal=new_signal)
    expected = {
        "description -> Cats and Dogs",
        "cat1 -> meow",
        "cat2 -> mrow",
        "dog1 -> woof",
        "dog2 -> arf",
        "dog3 -> bark",
        "dog4 -> ruff",
    }
    assert set(dc.collect_one("signal")) == expected
