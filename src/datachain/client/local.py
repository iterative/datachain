import os
import posixpath
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fsspec.implementations.local import LocalFileSystem

from datachain.lib.file import File

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.dataset import StorageURI


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
    def get_uri(cls, name) -> "StorageURI":
        from datachain.dataset import StorageURI

        return StorageURI(f'{cls.PREFIX}/{name.removeprefix("/")}')

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
        parsed = urlparse(url)
        if parsed.scheme == "file":
            scheme, rest = url.split(":", 1)
            url = f"{scheme.lower()}:{rest}"
        else:
            url = cls.path_to_uri(url)

        fill_path = url[len(cls.PREFIX) :]
        path_split = fill_path.rsplit("/", 1)
        bucket = path_split[0]
        if os.name == "nt":
            bucket = bucket.removeprefix("/")
        path = path_split[1] if len(path_split) > 1 else ""
        return bucket, path

    @classmethod
    def from_name(cls, name: str, cache, kwargs) -> "FileClient":
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

    async def get_current_etag(self, file: "File") -> str:
        info = self.fs.info(self.get_full_path(file.path))
        return self.info_to_file(info, "").etag

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

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            size=v.get("size", ""),
            etag=v["mtime"].hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(v["mtime"], timezone.utc),
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
