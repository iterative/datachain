from datetime import datetime, timezone
from typing import Any

from fsspec.implementations.memory import MemoryFileSystem

from datachain.lib.file import File
from datachain.node import Entry

from .fsspec import Client


class MemoryClient(Client):
    FS_CLASS = MemoryFileSystem
    PREFIX = "memory://"
    protocol = "memory"

    def convert_info(self, v: dict[str, Any], path: str) -> Entry:
        return Entry.from_file(
            path=path,
            size=v["size"],
            last_modified=datetime.fromtimestamp(v["created"], timezone.utc),
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        return File(
            path=path,
            size=v["size"],
            last_modified=datetime.fromtimestamp(v["created"], timezone.utc),
        )

    async def ls_dir(self, path):
        return self.fs.ls(path, detail=True)

    def rel_path(self, path):
        return path

    async def get_file(self, lpath, rpath, callback):
        return self.fs.get_file(lpath, rpath, callback=callback)
