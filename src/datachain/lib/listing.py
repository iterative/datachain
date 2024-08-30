from collections.abc import Iterator
from typing import Callable

from fsspec.asyn import get_loop

from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.lib.file import File


def list_bucket(uri: str, client_config=None) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        config = client_config or {}
        client, path = Client.parse_url(uri, None, **config)  # type: ignore[arg-type]
        for entries in iter_over_async(client.scandir(path), get_loop()):
            for entry in entries:
                yield entry.to_file(client.uri)

    return list_func
