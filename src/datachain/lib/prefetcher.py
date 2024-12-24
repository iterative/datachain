from collections.abc import Generator, Iterable, Sequence
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast

from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from datachain.asyn import AsyncMapper
from datachain.cache import temporary_cache
from datachain.lib.file import File

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from datachain.cache import DataChainCache as Cache
    from datachain.catalog.catalog import Catalog


T = TypeVar("T", bound=Sequence[Any])


def noop(*args, **kwargs):
    pass


async def _prefetch_input(row: T, catalog: "Catalog", download_cb: Callback) -> T:
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


def clone_catalog_with_cache(catalog: "Catalog", cache: "Cache") -> "Catalog":
    clone = catalog.copy()
    clone.cache = cache
    return clone


def rows_prefetcher(
    catalog: "Catalog",
    rows: Iterable[T],
    prefetch: int,
    cache: Optional["Cache"] = None,
    download_cb: Callback = DEFAULT_CALLBACK,
) -> Generator[T, None, None]:
    cache_ctx: AbstractContextManager[Cache]
    if cache:
        cache_ctx = nullcontext(cache)
    else:
        tmp_dir = catalog.cache.tmp_dir
        assert tmp_dir
        cache_ctx = temporary_cache(tmp_dir, prefix="prefetch-")

    with cache_ctx as prefetch_cache:
        catalog = clone_catalog_with_cache(catalog, prefetch_cache)
        func = partial(_prefetch_input, download_cb=download_cb, catalog=catalog)
        mapper = AsyncMapper(func, rows, workers=prefetch)
        yield from cast("Generator[T, None, None]", mapper.iterate())
