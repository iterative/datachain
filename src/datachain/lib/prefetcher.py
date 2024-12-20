import os
from contextlib import closing, contextmanager
from functools import partial

from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from datachain.asyn import AsyncMapper
from datachain.cache import DataChainCache
from datachain.lib.file import File


def noop(*args, **kwargs):
    pass


async def _prefetch_input(row, catalog, download_cb):
    try:
        callback = download_cb.increment_file_count
    except AttributeError:
        callback = noop

    for obj in row:
        if isinstance(obj, File):
            obj._set_stream(catalog, True, download_cb)
            await obj._prefetch()
            callback()
    return row


@contextmanager
def catalog_with_cache(catalog, cache):
    if not cache:
        yield
        return

    ocache = catalog.cache
    try:
        catalog.cache = cache
        yield
    finally:
        catalog.cache = ocache


def rows_prefetcher(
    catalog, path, rows, prefetch: int, download_cb: Callback = DEFAULT_CALLBACK
):
    cache = DataChainCache(
        f"/tmp/datachain/{path}/cache/",  # noqa: S108
        f"/tmp/datachain/{path}/tmp/",  # noqa: S108
    )
    os.makedirs(cache.cache_dir, exist_ok=True)
    os.makedirs(cache.tmp_dir, exist_ok=True)
    with catalog_with_cache(catalog, cache):
        func = partial(_prefetch_input, download_cb=download_cb, catalog=catalog)
        mapper = AsyncMapper(func, rows, workers=prefetch)
        with closing(mapper.iterate()) as result:
            yield from result
