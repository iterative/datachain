import gc
import os

import pytest

from datachain.cache import DataChainCache
from datachain.lib.pytorch import PytorchDataset
from datachain.lib.settings import Settings


@pytest.mark.parametrize(
    "cache,prefetch", [(True, 0), (True, 10), (False, 10), (False, 0)]
)
def test_cache(catalog, cache, prefetch):
    settings = Settings(cache=cache, prefetch=prefetch)
    ds = PytorchDataset("fake", 1, catalog, dc_settings=settings)
    assert ds.cache == cache
    assert ds.prefetch == prefetch

    if cache or not prefetch:
        assert catalog.cache is ds._cache
        return

    assert catalog.cache is not ds._cache
    head, tail = os.path.split(ds._cache.cache_dir)
    assert head == catalog.cache.tmp_dir
    assert tail.startswith("prefetch-")


@pytest.mark.parametrize("cache", [True, False])
def test_close(mocker, catalog, cache):
    spy = mocker.spy(DataChainCache, "destroy")
    ds = PytorchDataset("fake", 1, catalog, dc_settings=Settings(cache=cache))

    ds.close()

    if cache:
        spy.assert_not_called()
    else:
        spy.assert_called_once()


@pytest.mark.parametrize("cache", [True, False])
def test_cache_is_destroyed_on_gc(mocker, catalog, cache):
    spy = mocker.patch.object(DataChainCache, "destroy")
    ds = PytorchDataset("fake", 1, catalog, dc_settings=Settings(cache=cache))

    del ds
    gc.collect()

    if cache:
        spy.assert_not_called()
    else:
        spy.assert_called_once()
