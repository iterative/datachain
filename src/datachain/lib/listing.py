import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Callable, Optional

from botocore.exceptions import ClientError
from fsspec.asyn import get_loop

from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.error import ClientError as DataChainClientError
from datachain.lib.file import File

ResultQueue = asyncio.Queue[Optional[Sequence[File]]]

DELIMITER = "/"  # Path delimiter
FETCH_WORKERS = 100


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


def list_bucket(uri: str, client_config=None, fetch_workers=FETCH_WORKERS) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        config = client_config or {}
        client, path = Client.parse_url(uri, None, **config)  # type: ignore[arg-type]
        yield from iter_over_async(_scandir(client, path, fetch_workers), get_loop())

    return list_func
