import os
from typing import TYPE_CHECKING, Optional

from dvc_data.hashfile.db.local import LocalHashFileDB
from dvc_objects.fs.local import LocalFileSystem
from fsspec.callbacks import Callback, TqdmCallback

from .progress import Tqdm

if TYPE_CHECKING:
    from datachain.client import Client
    from datachain.lib.file import File


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

    def get_path(self, file: "File") -> Optional[str]:
        if self.contains(file):
            return self.path_from_checksum(file.get_hash())
        return None

    def contains(self, file: "File") -> bool:
        return self.odb.exists(file.get_hash())

    def path_from_checksum(self, checksum: str) -> str:
        assert checksum
        return self.odb.oid_to_path(checksum)

    def remove(self, file: "File") -> None:
        self.odb.delete(file.get_hash())

    async def download(
        self, file: "File", client: "Client", callback: Optional[Callback] = None
    ) -> None:
        from_path = f"{file.source}/{file.path}"
        from dvc_objects.fs.utils import tmp_fname

        odb_fs = self.odb.fs
        tmp_info = odb_fs.join(self.odb.tmp_dir, tmp_fname())  # type: ignore[arg-type]
        size = file.size
        if size < 0:
            size = await client.get_size(from_path, version_id=file.version)
        cb = callback or TqdmCallback(
            tqdm_kwargs={"desc": odb_fs.name(from_path), "bytes": True},
            tqdm_cls=Tqdm,
            size=size,
        )
        try:
            await client.get_file(
                from_path, tmp_info, callback=cb, version_id=file.version
            )
        finally:
            if not callback:
                cb.close()

        try:
            oid = file.get_hash()
            self.odb.add(tmp_info, self.odb.fs, oid)
        finally:
            os.unlink(tmp_info)

    def store_data(self, file: "File", contents: bytes) -> None:
        checksum = file.get_hash()
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
