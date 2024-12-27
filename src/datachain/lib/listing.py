import posixpath
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from fsspec.asyn import get_loop
from sqlalchemy.sql.expression import true

from datachain.asyn import iter_over_async
from datachain.client import Client
from datachain.lib.file import File
from datachain.query.schema import Column
from datachain.sql.functions import path as pathfunc
from datachain.telemetry import telemetry
from datachain.utils import uses_glob

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain
    from datachain.query.session import Session

LISTING_TTL = 4 * 60 * 60  # cached listing lasts 4 hours
LISTING_PREFIX = "lst__"  # listing datasets start with this name

D = TypeVar("D", bound="DataChain")


def list_bucket(uri: str, cache, client_config=None) -> Callable:
    """
    Function that returns another generator function that yields File objects
    from bucket where each File represents one bucket entry.
    """

    def list_func() -> Iterator[File]:
        config = client_config or {}
        client = Client.get_client(uri, cache, **config)  # type: ignore[arg-type]
        _, path = Client.parse_url(uri)
        for entries in iter_over_async(client.scandir(path.rstrip("/")), get_loop()):
            yield from entries

    return list_func


def get_file_info(uri: str, cache, client_config=None) -> File:
    """
    Wrapper to return File object by its URI
    """
    client = Client.get_client(uri, cache, **(client_config or {}))  # type: ignore[arg-type]
    _, path = Client.parse_url(uri)
    return client.get_file_info(path)


def ls(
    dc: D,
    path: str,
    recursive: Optional[bool] = True,
    object_name="file",
) -> D:
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


def parse_listing_uri(uri: str, cache, client_config) -> tuple[Optional[str], str, str]:
    """
    Parsing uri and returns listing dataset name, listing uri and listing path
    """
    client_config = client_config or {}
    client = Client.get_client(uri, cache, **client_config)
    storage_uri, path = Client.parse_url(uri)
    telemetry.log_param("client", client.PREFIX)

    if not uri.endswith("/") and client.fs.isfile(uri):
        return None, f'{storage_uri}/{path.lstrip("/")}', path
    if uses_glob(path):
        lst_uri_path = posixpath.dirname(path)
    else:
        storage_uri, path = Client.parse_url(f'{uri.rstrip("/")}/')
        lst_uri_path = path

    lst_uri = f'{storage_uri}/{lst_uri_path.lstrip("/")}'
    ds_name = (
        f"{LISTING_PREFIX}{storage_uri}/{posixpath.join(lst_uri_path, '').lstrip('/')}"
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


def get_listing(
    uri: str, session: "Session", update: bool = False
) -> tuple[Optional[str], str, str, bool]:
    """Returns correct listing dataset name that must be used for saving listing
    operation. It takes into account existing listings and reusability of those.
    It also returns boolean saying if returned dataset name is reused / already
    exists or not (on update it always returns False - just because there was no
    reason to complicate it so far). And it returns correct listing path that should
    be used to find rows based on uri.
    """
    from datachain.client.local import FileClient

    catalog = session.catalog
    cache = catalog.cache
    client_config = catalog.client_config

    client = Client.get_client(uri, cache, **client_config)
    ds_name, list_uri, list_path = parse_listing_uri(uri, cache, client_config)
    listing = None

    # if we don't want to use cached dataset (e.g. for a single file listing)
    if not ds_name:
        return None, list_uri, list_path, False

    listings = [
        ls for ls in catalog.listings() if not ls.is_expired and ls.contains(ds_name)
    ]

    # if no need to update - choosing the most recent one;
    # otherwise, we'll using the exact original `ds_name`` in this case:
    # - if a "bigger" listing exists, we don't want to update it, it's better
    #   to create a new "smaller" one on "update=True"
    # - if an exact listing exists it will have the same name as `ds_name`
    #   anyway below
    if listings and not update:
        listing = sorted(listings, key=lambda ls: ls.created_at)[-1]

    # for local file system we need to fix listing path / prefix
    # if we are reusing existing listing
    if isinstance(client, FileClient) and listing and listing.name != ds_name:
        list_path = f'{ds_name.strip("/").removeprefix(listing.name)}/{list_path}'

    ds_name = listing.name if listing else ds_name

    return ds_name, list_uri, list_path, bool(listing)
