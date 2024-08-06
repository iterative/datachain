import hashlib
import json
import os
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Optional

import attrs
from dvc_data.hashfile.db.local import LocalHashFileDB
from dvc_objects.fs.local import LocalFileSystem
from fsspec.callbacks import Callback, TqdmCallback

from datachain.utils import TIME_ZERO

from .progress import Tqdm

if TYPE_CHECKING:
    from datachain.client import Client
    from datachain.storage import StorageURI

sha256 = partial(hashlib.sha256, usedforsecurity=False)


@attrs.frozen
class UniqueId:
    storage: "StorageURI"
    path: str
    size: int
    etag: str
    version: str = ""
    is_latest: bool = True
    vtype: str = ""
    location: Optional[str] = None
    last_modified: datetime = TIME_ZERO

    def get_parsed_location(self) -> Optional[dict]:
        if not self.location:
            return None

        loc_stack = (
            json.loads(self.location)
            if isinstance(self.location, str)
            else self.location
        )
        if len(loc_stack) > 1:
            raise NotImplementedError("Nested v-objects are not supported yet.")

        return loc_stack[0]

    def get_hash(self) -> str:
        fingerprint = f"{self.storage}/{self.path}/{self.version}/{self.etag}"
        if self.location:
            fingerprint += f"/{self.location}"
        return sha256(fingerprint.encode()).hexdigest()


def try_scandir(path):
    try:
        with os.scandir(path) as it:
            yield from it
    except OSError:
        pass


class DataChainCache:
    def __init__(self, cache_dir: str, tmp_dir: str):
        self.odb = LocalHashFileDB(
            LocalFileSystem(),
            cache_dir,
            tmp_dir=tmp_dir,
        )

    @property
    def cache_dir(self):
        return self.odb.path

    @property
    def tmp_dir(self):
        return self.odb.tmp_dir

    def get_path(self, uid: UniqueId) -> Optional[str]:
        if self.contains(uid):
            return self.path_from_checksum(uid.get_hash())
        return None

    def contains(self, uid: UniqueId) -> bool:
        return self.odb.exists(uid.get_hash())

    def path_from_checksum(self, checksum: str) -> str:
        assert checksum
        return self.odb.oid_to_path(checksum)

    def remove(self, uid: UniqueId) -> None:
        self.odb.delete(uid.get_hash())

    async def download(
        self, uid: UniqueId, client: "Client", callback: Optional[Callback] = None
    ) -> None:
        from_path = f"{uid.storage}/{uid.path}"
        from dvc_objects.fs.utils import tmp_fname

        odb_fs = self.odb.fs
        tmp_info = odb_fs.join(self.odb.tmp_dir, tmp_fname())  # type: ignore[arg-type]
        size = uid.size
        if size < 0:
            size = await client.get_size(from_path)
        cb = callback or TqdmCallback(
            tqdm_kwargs={"desc": odb_fs.name(from_path), "bytes": True},
            tqdm_cls=Tqdm,
            size=size,
        )
        try:
            await client.get_file(from_path, tmp_info, callback=cb)
        finally:
            if not callback:
                cb.close()

        try:
            oid = uid.get_hash()
            self.odb.add(tmp_info, self.odb.fs, oid)
        finally:
            os.unlink(tmp_info)

    def store_data(self, uid: UniqueId, contents: bytes) -> None:
        checksum = uid.get_hash()
        dst = self.path_from_checksum(checksum)
        if not os.path.exists(dst):
            # Create the file only if it's not already in cache
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, mode="wb") as f:
                f.write(contents)

    def clear(self):
        """
        Completely clear the cache.
        """
        self.odb.clear()

    def get_total_size(self) -> int:
        total = 0
        for subdir in try_scandir(self.odb.path):
            for file in try_scandir(subdir):
                try:
                    total += file.stat().st_size
                except OSError:
                    pass
        return total
