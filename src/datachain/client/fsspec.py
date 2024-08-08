import asyncio
import functools
import logging
import multiprocessing
import os
import posixpath
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from datetime import datetime
from shutil import copy2
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
    NamedTuple,
    Optional,
)
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from dvc_objects.fs.system import reflink
from fsspec.asyn import get_loop, sync
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from tqdm import tqdm

from datachain.cache import DataChainCache, UniqueId
from datachain.client.fileslice import FileSlice, FileWrapper
from datachain.error import ClientError as DataChainClientError
from datachain.node import Entry
from datachain.nodes_fetcher import NodesFetcher
from datachain.nodes_thread_pool import NodeChunk
from datachain.storage import StorageURI

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem


logger = logging.getLogger("datachain")

FETCH_WORKERS = 100
DELIMITER = "/"  # Path delimiter.

DATA_SOURCE_URI_PATTERN = re.compile(r"^[\w]+:\/\/.*$")

ResultQueue = asyncio.Queue[Optional[Sequence[Entry]]]


def _is_win_local_path(uri: str) -> bool:
    if sys.platform == "win32":
        if len(uri) >= 1 and uri[0] == "\\":
            return True
        if (
            len(uri) >= 3
            and uri[1] == ":"
            and (uri[2] == "/" or uri[2] == "\\")
            and uri[0].isalpha()
        ):
            return True
    return False


class Bucket(NamedTuple):
    name: str
    uri: StorageURI
    created: Optional[datetime]


class Client(ABC):
    MAX_THREADS = multiprocessing.cpu_count()
    FS_CLASS: ClassVar[type["AbstractFileSystem"]]
    PREFIX: ClassVar[str]
    protocol: ClassVar[str]

    def __init__(
        self, name: str, fs_kwargs: dict[str, Any], cache: DataChainCache
    ) -> None:
        self.name = name
        self.fs_kwargs = fs_kwargs
        self._fs: Optional[AbstractFileSystem] = None
        self.cache = cache
        self.uri = self.get_uri(self.name)

    @staticmethod
    def get_implementation(url: str) -> type["Client"]:
        from .azure import AzureClient
        from .gcs import GCSClient
        from .local import FileClient
        from .s3 import ClientS3

        protocol = urlparse(url).scheme

        if not protocol or _is_win_local_path(url):
            return FileClient

        protocol = protocol.lower()
        if protocol == ClientS3.protocol:
            return ClientS3
        if protocol == GCSClient.protocol:
            return GCSClient
        if protocol == AzureClient.protocol:
            return AzureClient
        if protocol == FileClient.protocol:
            return FileClient

        raise NotImplementedError(f"Unsupported protocol: {protocol}")

    @staticmethod
    def is_data_source_uri(name: str) -> bool:
        # Returns True if name is one of supported data sources URIs, e.g s3 bucket
        return DATA_SOURCE_URI_PATTERN.match(name) is not None

    @staticmethod
    def parse_url(
        source: str,
        cache: DataChainCache,
        **kwargs,
    ) -> tuple["Client", str]:
        cls = Client.get_implementation(source)
        storage_url, rel_path = cls.split_url(source)
        client = cls.from_name(storage_url, cache, kwargs)
        return client, rel_path

    @classmethod
    def create_fs(cls, **kwargs) -> "AbstractFileSystem":
        kwargs.setdefault("version_aware", True)
        fs = cls.FS_CLASS(**kwargs)
        fs.invalidate_cache()
        return fs

    @classmethod
    def from_name(
        cls,
        name: str,
        cache: DataChainCache,
        kwargs: dict[str, Any],
    ) -> "Client":
        return cls(name, kwargs, cache)

    @classmethod
    def from_source(
        cls,
        uri: StorageURI,
        cache: DataChainCache,
        **kwargs,
    ) -> "Client":
        return cls(cls.FS_CLASS._strip_protocol(uri), kwargs, cache)

    @classmethod
    def ls_buckets(cls, **kwargs) -> Iterator[Bucket]:
        for entry in cls.create_fs(**kwargs).ls(cls.PREFIX, detail=True):
            name = entry["name"].rstrip("/")
            yield Bucket(
                name=name,
                uri=StorageURI(f"{cls.PREFIX}{name}"),
                created=entry.get("CreationDate"),
            )

    @classmethod
    def is_root_url(cls, url) -> bool:
        return url == cls.PREFIX

    @classmethod
    def get_uri(cls, name) -> StorageURI:
        return StorageURI(f"{cls.PREFIX}{name}")

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        fill_path = url[len(cls.PREFIX) :]
        path_split = fill_path.split("/", 1)
        bucket = path_split[0]
        path = path_split[1] if len(path_split) > 1 else ""
        return bucket, path

    @property
    def fs(self) -> "AbstractFileSystem":
        if not self._fs:
            self._fs = self.create_fs(**self.fs_kwargs)
        return self._fs

    def url(self, path: str, expires: int = 3600, **kwargs) -> str:
        return self.fs.sign(self.get_full_path(path), expiration=expires, **kwargs)

    async def get_current_etag(self, uid: UniqueId) -> str:
        info = await self.fs._info(self.get_full_path(uid.path))
        return self.convert_info(info, "").etag

    async def get_size(self, path: str) -> int:
        return await self.fs._size(path)

    async def get_file(self, lpath, rpath, callback):
        return await self.fs._get_file(lpath, rpath, callback=callback)

    async def scandir(
        self, start_prefix: str, method: str = "default"
    ) -> AsyncIterator[Sequence[Entry]]:
        try:
            impl = getattr(self, f"_fetch_{method}")
        except AttributeError:
            raise ValueError(f"Unknown indexing method '{method}'") from None
        result_queue: ResultQueue = asyncio.Queue()
        loop = get_loop()
        main_task = loop.create_task(impl(start_prefix, result_queue))
        while (entry := await result_queue.get()) is not None:
            yield entry
        await main_task

    async def _fetch_nested(self, start_prefix: str, result_queue: ResultQueue) -> None:
        progress_bar = tqdm(desc=f"Listing {self.uri}", unit=" objects")
        loop = get_loop()

        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait(start_prefix)

        async def process(queue) -> None:
            while True:
                prefix = await queue.get()
                try:
                    subdirs = await self._fetch_dir(prefix, progress_bar, result_queue)
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
                loop.create_task(process(queue)) for _ in range(FETCH_WORKERS)
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
            progress_bar.close()
            result_queue.put_nowait(None)

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        await self._fetch_nested(start_prefix, result_queue)

    async def _fetch_dir(self, prefix, pbar, result_queue) -> set[str]:
        path = f"{self.name}/{prefix}"
        infos = await self.ls_dir(path)
        files = []
        subdirs = set()
        for info in infos:
            full_path = info["name"]
            subprefix = self.rel_path(full_path)
            if prefix.strip(DELIMITER) == subprefix.strip(DELIMITER):
                continue
            if info["type"] == "directory":
                subdirs.add(subprefix)
            else:
                files.append(self.convert_info(info, subprefix))
        if files:
            await result_queue.put(files)
        found_count = len(subdirs) + len(files)
        pbar.update(found_count)
        return subdirs

    @staticmethod
    def _is_valid_key(key: str) -> bool:
        """
        Check if the key looks like a valid path.

        Invalid keys are ignored when indexing.
        """
        return not (key.startswith("/") or key.endswith("/") or "//" in key)

    async def ls_dir(self, path):
        return await self.fs._ls(path, detail=True, versions=True)

    def rel_path(self, path: str) -> str:
        return self.fs.split_path(path)[1]

    def get_full_path(self, rel_path: str) -> str:
        return f"{self.PREFIX}{self.name}/{rel_path}"

    @abstractmethod
    def convert_info(self, v: dict[str, Any], parent: str) -> Entry: ...

    def fetch_nodes(
        self,
        nodes,
        shared_progress_bar=None,
    ) -> None:
        fetcher = NodesFetcher(self, self.MAX_THREADS, self.cache)
        chunk_gen = NodeChunk(self.cache, self.uri, nodes)
        fetcher.run(chunk_gen, shared_progress_bar)

    def instantiate_object(
        self,
        uid: UniqueId,
        dst: str,
        progress_bar: tqdm,
        force: bool = False,
    ) -> None:
        if os.path.exists(dst):
            if force:
                os.remove(dst)
            else:
                progress_bar.close()
                raise FileExistsError(f"Path {dst} already exists")
        self.do_instantiate_object(uid, dst)

    def do_instantiate_object(self, uid: "UniqueId", dst: str) -> None:
        src = self.cache.get_path(uid)
        assert src is not None

        try:
            reflink(src, dst)
        except OSError:
            # Default to copy if reflinks are not supported
            copy2(src, dst)

    def open_object(
        self, uid: UniqueId, use_cache: bool = True, cb: Callback = DEFAULT_CALLBACK
    ) -> BinaryIO:
        """Open a file, including files in tar archives."""
        location = uid.get_parsed_location()
        if use_cache and (cache_path := self.cache.get_path(uid)):
            return open(cache_path, mode="rb")  # noqa: SIM115
        if location and location["vtype"] == "tar":
            return self._open_tar(uid, use_cache=True)
        return FileWrapper(self.fs.open(self.get_full_path(uid.path)), cb)  # type: ignore[return-value]

    def _open_tar(self, uid: UniqueId, use_cache: bool = True):
        location = uid.get_parsed_location()
        assert location

        offset = location["offset"]
        size = location["size"]
        parent = location["parent"]

        parent_uid = UniqueId(
            parent["source"],
            parent["path"],
            parent["size"],
            parent["etag"],
            vtype=parent["vtype"],
            location=parent["location"],
        )
        f = self.open_object(parent_uid, use_cache=use_cache)
        return FileSlice(f, offset, size, posixpath.basename(uid.path))

    def download(self, uid: UniqueId, *, callback: Callback = DEFAULT_CALLBACK) -> None:
        sync(get_loop(), functools.partial(self._download, uid, callback=callback))

    async def _download(self, uid: UniqueId, *, callback: "Callback" = None) -> None:
        if self.cache.contains(uid):
            # Already in cache, so there's nothing to do.
            return
        await self._put_in_cache(uid, callback=callback)

    def put_in_cache(self, uid: UniqueId, *, callback: "Callback" = None) -> None:
        sync(get_loop(), functools.partial(self._put_in_cache, uid, callback=callback))

    async def _put_in_cache(
        self, uid: UniqueId, *, callback: "Callback" = None
    ) -> None:
        location = uid.get_parsed_location()
        if location and location["vtype"] == "tar":
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, functools.partial(self._download_from_tar, uid, callback=callback)
            )
            return
        if uid.etag:
            etag = await self.get_current_etag(uid)
            if uid.etag != etag:
                raise FileNotFoundError(
                    f"Invalid etag for {uid.storage}/{uid.path}: "
                    f"expected {uid.etag}, got {etag}"
                )
        await self.cache.download(uid, self, callback=callback)

    def _download_from_tar(self, uid, *, callback: "Callback" = None):
        with self._open_tar(uid, use_cache=False) as f:
            contents = f.read()
        self.cache.store_data(uid, contents)
