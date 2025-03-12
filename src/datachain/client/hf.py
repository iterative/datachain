import functools
import posixpath
from typing import Any

from datachain.lib.file import File

from .fsspec import Client


class classproperty:  # noqa: N801
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


@functools.cache
def get_hf_filesystem_cls():
    import fsspec
    from packaging.version import Version, parse

    fsspec_version = parse(fsspec.__version__)
    minver = Version("2024.12.0")

    if fsspec_version < minver:
        raise ImportError(
            f"datachain requires 'fsspec>={minver}' but version "
            f"{fsspec_version} is installed."
        )

    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from huggingface_hub import HfFileSystem

    fs_cls = AsyncFileSystemWrapper.wrap_class(HfFileSystem)
    # AsyncFileSystemWrapper does not set class properties, so we need to set them back.
    fs_cls.protocol = HfFileSystem.protocol
    return fs_cls


class HfClient(Client):
    PREFIX = "hf://"
    protocol = "hf"

    @classproperty
    def FS_CLASS(cls):  # noqa: N802, N805
        return get_hf_filesystem_cls()

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            size=v["size"],
            version=v["last_commit"].oid,
            etag=v.get("blob_id", ""),
            last_modified=v["last_commit"].date,
        )

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
