import posixpath
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

from fsspec.asyn import get_loop
from sqlalchemy.sql.expression import true

from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.lib.file import File
from datachain.query.schema import Column
from datachain.sql.functions import path as pathfunc
from datachain.utils import uses_glob

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain

LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name


def list_bucket(uri: str, client_config=None) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        config = client_config or {}
        client, path = Client.parse_url(uri, None, **config)  # type: ignore[arg-type]
        for entries in iter_over_async(client.scandir(path.rstrip("/")), get_loop()):
            for entry in entries:
                yield entry.to_file(client.uri)

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


def listing_uri_from_name(dataset_name: str) -> str:
    """Returns clean storage URI from listing dataset name"""
    if not is_listing_dataset(dataset_name):
        raise ValueError(f"Dataset {dataset_name} is not a listing")
    return dataset_name.removeprefix(LISTING_PREFIX)


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
