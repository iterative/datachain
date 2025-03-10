import os
import posixpath
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from huggingface_hub import HfFileSystem

from datachain.lib.file import File

from .fsspec import Client

AsyncHfFileSystem = AsyncFileSystemWrapper.wrap_class(HfFileSystem)
# AsyncFileSystemWrapper does not set class properties, so we need to set them back.
AsyncHfFileSystem.protocol = HfFileSystem.protocol


class HfClient(Client):
    FS_CLASS = AsyncHfFileSystem
    PREFIX = "hf://"
    protocol = "hf"

    @classmethod
    def create_fs(cls, **kwargs) -> AbstractFileSystem:
        if os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]

        return super().create_fs(**kwargs)

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            source=self.uri,
            path=path,
            size=v["size"],
            version=v["last_commit"].oid,
            etag=v.get("blob_id", ""),
            last_modified=v["last_commit"].date,
        )

    async def ls_dir(self, path):
        return await self.fs._ls(path, detail=True)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
