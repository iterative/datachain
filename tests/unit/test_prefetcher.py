import os

from datachain.cache import DataChainCache, get_temp_cache
from datachain.lib.file import File
from datachain.lib.prefetcher import rows_prefetcher


def test_prefetcher(mocker, tmp_dir, catalog):
    rows = []
    for path in "abcd":
        (tmp_dir / path).write_text(path)
        row = (File(path=str(tmp_dir / path)),)
        rows.append(row)

    for (file,) in rows_prefetcher(catalog, rows, prefetch=5):
        assert file._catalog
        head, tail = os.path.split(file._catalog.cache.cache_dir)
        assert head == catalog.cache.tmp_dir
        assert tail.startswith("prefetch-")
        assert file._catalog.cache.contains(file)

    cache = get_temp_cache(tmp_dir)
    for (file,) in rows_prefetcher(catalog, rows, prefetch=5, cache=cache):
        assert file._catalog
        assert file._catalog.cache == cache
        assert cache.contains(file)


def test_prefetcher_closes_temp_cache(mocker, tmp_dir, catalog):
    rows = []
    for path in "abcd":
        (tmp_dir / path).write_text(path)
        row = (File(path=str(tmp_dir / path)),)
        rows.append(row)
    spy = mocker.spy(DataChainCache, "destroy")

    rows_gen = rows_prefetcher(catalog, rows, prefetch=5)
    next(rows_gen)
    rows_gen.close()
    assert spy.called
