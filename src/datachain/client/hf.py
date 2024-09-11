import os
import posixpath
from typing import Any, cast

from huggingface_hub import HfFileSystem

from datachain.lib.file import File
from datachain.node import Entry

from .fsspec import Client


class HfClient(Client):
    FS_CLASS = HfFileSystem
    PREFIX = "hf://"
    protocol = "hf"

    @classmethod
    def create_fs(cls, **kwargs) -> HfFileSystem:
        if os.environ.get("HF_TOKEN"):
            kwargs["token"] = os.environ["HF_TOKEN"]

        return cast(HfFileSystem, super().create_fs(**kwargs))

    def convert_info(self, v: dict[str, Any], path: str) -> Entry:
        return Entry.from_file(
            path=path,
            size=v["size"],
            version=v["last_commit"].oid,
            etag=v.get("blob_id", ""),
            last_modified=v["last_commit"].date,
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            path=path,
            size=v["size"],
            version=v["last_commit"].oid,
            etag=v.get("blob_id", ""),
            last_modified=v["last_commit"].date,
        )

    async def ls_dir(self, path):
        return self.fs.ls(path, detail=True)

    def rel_path(self, path):
        return posixpath.relpath(path, self.name)
