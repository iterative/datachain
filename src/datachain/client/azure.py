from typing import Any

from adlfs import AzureBlobFileSystem
from tqdm import tqdm

from datachain.lib.file import File
from datachain.node import Entry

from .fsspec import DELIMITER, Client, ResultQueue


class AzureClient(Client):
    FS_CLASS = AzureBlobFileSystem
    PREFIX = "az://"
    protocol = "az"

    def convert_info(self, v: dict[str, Any], path: str) -> Entry:
        version_id = v.get("version_id")
        return Entry.from_file(
            path=path,
            etag=v.get("etag", "").strip('"'),
            version=version_id or "",
            is_latest=version_id is None or bool(v.get("is_current_version")),
            last_modified=v["last_modified"],
            size=v.get("size", ""),
        )

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        version_id = v.get("version_id")
        return File(
            source=self.uri,
            path=path,
            etag=v.get("etag", "").strip('"'),
            version=version_id or "",
            is_latest=version_id is None or bool(v.get("is_current_version")),
            last_modified=v["last_modified"],
            size=v.get("size", ""),
        )

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        prefix = start_prefix
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        found = False
        try:
            with tqdm(desc=f"Listing {self.uri}", unit=" objects") as pbar:
                async with self.fs.service_client.get_container_client(
                    container=self.name
                ) as container_client:
                    async for page in container_client.list_blobs(
                        include=["metadata", "versions"], name_starts_with=prefix
                    ).by_page():
                        entries = []
                        async for b in page:
                            found = True
                            if not self._is_valid_key(b["name"]):
                                continue
                            info = (await self.fs._details([b]))[0]
                            entries.append(
                                self.convert_info(info, self.rel_path(info["name"]))
                            )
                        if entries:
                            await result_queue.put(entries)
                            pbar.update(len(entries))
                    if not found:
                        raise FileNotFoundError(
                            f"Unable to resolve remote path: {prefix}"
                        )
        finally:
            result_queue.put_nowait(None)

    _fetch_default = _fetch_flat
