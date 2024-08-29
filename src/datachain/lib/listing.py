import asyncio
import posixpath
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

from botocore.exceptions import ClientError
from fsspec.asyn import get_loop
from sqlalchemy.sql.expression import true

from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.error import ClientError as DataChainClientError
from datachain.lib.file import File
from datachain.query.schema import Column
from datachain.sql.functions import path as pathfunc
from datachain.utils import uses_glob

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


ResultQueue = asyncio.Queue[Optional[Sequence[File]]]

DELIMITER = "/"  # Path delimiter
FETCH_WORKERS = 100
LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name


async def _fetch_dir(client, prefix, result_queue) -> set[str]:
    path = f"{client.name}/{prefix}"
    infos = await client.ls_dir(path)
    files = []
    subdirs = set()
    for info in infos:
        full_path = info["name"]
        subprefix = client.rel_path(full_path)
        if prefix.strip(DELIMITER) == subprefix.strip(DELIMITER):
            continue
        if info["type"] == "directory":
            subdirs.add(subprefix)
        else:
            files.append(client.info_to_file(info, subprefix))
    if files:
        await result_queue.put(files)
    return subdirs


async def _fetch(
    client, start_prefix: str, result_queue: ResultQueue, fetch_workers
) -> None:
    loop = get_loop()

    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait(start_prefix)

    async def process(queue) -> None:
        while True:
            prefix = await queue.get()
            try:
                subdirs = await _fetch_dir(client, prefix, result_queue)
                for subdir in subdirs:
                    queue.put_nowait(subdir)
            except Exception:
                while not queue.empty():
                    queue.get_nowait()
                    queue.task_done()
                raise

            finally:
                queue.task_done()

    try:
        workers: list[asyncio.Task] = [
            loop.create_task(process(queue)) for _ in range(fetch_workers)
        ]

        # Wait for all fetch tasks to complete
        await queue.join()
        # Stop the workers
        excs = []
        for worker in workers:
            if worker.done() and (exc := worker.exception()):
                excs.append(exc)
            else:
                worker.cancel()
        if excs:
            raise excs[0]
    except ClientError as exc:
        raise DataChainClientError(
            exc.response.get("Error", {}).get("Message") or exc,
            exc.response.get("Error", {}).get("Code"),
        ) from exc
    finally:
        # This ensures the progress bar is closed before any exceptions are raised
        result_queue.put_nowait(None)


async def _scandir(client, prefix, fetch_workers) -> AsyncIterator:
    """Recursively goes through dir tree and yields files"""
    result_queue: ResultQueue = asyncio.Queue()
    loop = get_loop()
    main_task = loop.create_task(_fetch(client, prefix, result_queue, fetch_workers))
    while (files := await result_queue.get()) is not None:
        for f in files:
            yield f

    await main_task


def list_bucket(uri: str, fetch_workers=FETCH_WORKERS, **kwargs) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        client, path = Client.parse_url(uri, None, **kwargs)  # type: ignore[arg-type]
        yield from iter_over_async(_scandir(client, path, fetch_workers), get_loop())

    return list_func


def ls(
    dc: "DataChain",
    path: str,
    recursive: Optional[bool] = True,
    object_name="file",
):
    """
    Return files by some path from DataChain instance which contains bucket listing.
    Path can have globs.
    If recursive is set to False, only first level children will be returned by
    specified path
    """

    def _file_c(name: str) -> Column:
        return Column(f"{object_name}.{name}")

    dc = dc.filter(_file_c("is_latest") == true())

    if recursive:
        if not path or path == "/":
            # root of a bucket, returning all latest files from it
            return dc

        if not uses_glob(path):
            # path is not glob, so it's pointing to some directory or a specific
            # file and we are adding proper filter for it
            return dc.filter(
                (_file_c("path") == path)
                | (_file_c("path").glob(path.rstrip("/") + "/*"))
            )

        # path has glob syntax so we are returning glob filter
        return dc.filter(_file_c("path").glob(path))
    # returning only first level children by path
    return dc.filter(pathfunc.parent(_file_c("path")) == path.lstrip("/").rstrip("/*"))


def parse_listing_uri(uri: str, cache, client_config) -> tuple[str, str, str]:
    """
    Parsing uri and returns listing dataset name, listing uri and listing path
    """
    client, path = Client.parse_url(uri, cache, **client_config)

    # clean path without globs
    lst_uri_path = (
        posixpath.dirname(path) if uses_glob(path) or client.fs.isfile(uri) else path
    )

    lst_uri = f"{client.uri}/{lst_uri_path.lstrip('/')}"
    ds_name = (
        f"{LISTING_PREFIX}{client.uri}/{posixpath.join(lst_uri_path, '').lstrip('/')}"
    )

    return ds_name, lst_uri, path


def is_listing_dataset(name: str) -> bool:
    """Returns True if it's special listing dataset"""
    return name.startswith(LISTING_PREFIX)


def is_listing_expired(created_at: datetime) -> bool:
    """Checks if listing has expired based on it's creation date"""
    return datetime.now(timezone.utc) > created_at + timedelta(seconds=LISTING_TTL)


def is_listing_subset(ds1_name: str, ds2_name: str) -> bool:
    """
    Checks if one listing contains another one by comparing corresponding dataset names
    """
    assert ds1_name.endswith("/")
    assert ds2_name.endswith("/")

    return ds2_name.startswith(ds1_name)
