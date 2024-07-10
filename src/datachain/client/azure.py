import posixpath
from typing import Any

from adlfs import AzureBlobFileSystem
from tqdm import tqdm

from datachain.node import Entry

from .fsspec import DELIMITER, Client, ResultQueue


class AzureClient(Client):
    FS_CLASS = AzureBlobFileSystem
    PREFIX = "az://"
    protocol = "az"

    def convert_info(self, v: dict[str, Any], parent: str) -> Entry:
        version_id = v.get("version_id")
        name = v.get("name", "").split(DELIMITER)[-1]
        if version_id:
            version_suffix = f"?versionid={version_id}"
            if name.endswith(version_suffix):
                name = name[: -len(version_suffix)]
        return Entry.from_file(
            parent=parent,
            name=name,
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
                            full_path = info["name"]
                            parent = posixpath.dirname(self.rel_path(full_path))
                            entries.append(self.convert_info(info, parent))
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
