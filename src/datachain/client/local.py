import os
import posixpath
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fsspec.implementations.local import LocalFileSystem

from datachain.node import Entry
from datachain.storage import StorageURI

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.data_storage import AbstractMetastore


class FileClient(Client):
    FS_CLASS = LocalFileSystem
    PREFIX = "file://"
    protocol = "file"

    def __init__(
        self, name: str, fs_kwargs: dict[str, Any], cache, use_symlinks: bool = False
    ) -> None:
        super().__init__(name, fs_kwargs, cache)
        self.use_symlinks = use_symlinks

    def url(self, path: str, expires: int = 3600, **kwargs) -> str:
        raise TypeError("Signed urls are not implemented for local file system")

    @classmethod
    def get_uri(cls, name) -> StorageURI:
        """
        This returns root of FS as uri, e.g
            Linux & Mac : file:///
            Windows: file:///C:/
        """
        return StorageURI(Path(name).as_uri())

    @staticmethod
    def root_dir() -> str:
        """
        Returns file system root path.
        Linux &  MacOS: /
        Windows: C:/
        """
        return Path.cwd().anchor.replace(os.sep, posixpath.sep)

    @staticmethod
    def root_path() -> Path:
        return Path(FileClient.root_dir())

    @classmethod
    def ls_buckets(cls, **kwargs):
        return []

    @classmethod
    def path_to_uri(cls, path: str) -> str:
        """
        Resolving path, that can be absolute or relative, to file URI which
        starts with file:/// prefix
        In unix like systems we support home shortcut as well.
        Examples:
            ./animals -> file:///home/user/working_dir/animals
            ~/animals -> file:///home/user/animals
            /home/user/animals -> file:///home/user/animals
            /home/user/animals/ -> file:///home/user/animals/
            C:\\windows\animals -> file:///C:/windows/animals
        """
        uri = Path(path).expanduser().absolute().resolve().as_uri()
        if path[-1] == os.sep:
            # we should keep os separator from the end of the path
            uri += "/"  # in uri (file:///...) all separators are / regardless of os

        return uri

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        """
        Splits url into two components:
            1. root of the FS which will later on become the name of the storage
            2. path which will later on become partial path
        Note that URL needs to be have file:/// protocol.
        Examples:
            file:///tmp/dir -> / + tmp/dir
            file:///c:/windows/files -> c:/ + windows/files
        """
        parsed = urlparse(url)
        if parsed.scheme == "file":
            scheme, rest = url.split(":", 1)
            uri = f"{scheme.lower()}:{rest}"
        else:
            uri = cls.path_to_uri(url)

        return cls.root_dir(), uri.removeprefix(cls.root_path().as_uri())

    @classmethod
    def from_name(
        cls, name: str, metastore: "AbstractMetastore", cache, kwargs
    ) -> "FileClient":
        use_symlinks = kwargs.pop("use_symlinks", False)
        return cls(name, kwargs, cache, use_symlinks=use_symlinks)

    @classmethod
    def from_source(
        cls,
        uri: str,
        cache,
        use_symlinks: bool = False,
        **kwargs,
    ) -> "FileClient":
        return cls(
            LocalFileSystem._strip_protocol(uri),
            kwargs,
            cache,
            use_symlinks=use_symlinks,
        )

    async def get_current_etag(self, uid) -> str:
        info = self.fs.info(self.get_full_path(uid.path))
        return self.convert_info(info, "").etag

    async def get_size(self, path: str) -> int:
        return self.fs.size(path)

    async def get_file(self, lpath, rpath, callback):
        return self.fs.get_file(lpath, rpath, callback=callback)

    async def ls_dir(self, path):
        return self.fs.ls(path, detail=True)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)

    def get_full_path(self, rel_path):
        full_path = Path(self.name, rel_path).as_posix()
        if rel_path.endswith("/") or not rel_path:
            full_path += "/"
        return full_path

    def convert_info(self, v: dict[str, Any], parent: str) -> Entry:
        name = posixpath.basename(v["name"])
        return Entry.from_file(
            parent=parent,
            name=name,
            etag=v["mtime"].hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(v["mtime"], timezone.utc),
            size=v.get("size", ""),
        )

    def fetch_nodes(
        self,
        nodes,
        shared_progress_bar=None,
    ) -> None:
        if not self.use_symlinks:
            super().fetch_nodes(nodes, shared_progress_bar)

    def do_instantiate_object(self, uid, dst):
        if self.use_symlinks:
            os.symlink(Path(self.name, uid.path), dst)
        else:
            super().do_instantiate_object(uid, dst)
