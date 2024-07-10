import asyncio
import json
import os
import posixpath
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional, cast

from dateutil.parser import isoparse
from gcsfs import GCSFileSystem
from tqdm import tqdm

from datachain.node import Entry

from .fsspec import DELIMITER, Client, ResultQueue

# Patch gcsfs for consistency with s3fs
GCSFileSystem.set_session = GCSFileSystem._set_session
PageQueue = asyncio.Queue[Optional[Iterable[dict[str, Any]]]]


class GCSClient(Client):
    FS_CLASS = GCSFileSystem
    PREFIX = "gs://"
    protocol = "gs"

    @classmethod
    def create_fs(cls, **kwargs) -> GCSFileSystem:
        if os.environ.get("DATACHAIN_GCP_CREDENTIALS"):
            kwargs["token"] = json.loads(os.environ["DATACHAIN_GCP_CREDENTIALS"])
        if kwargs.pop("anon", False):
            kwargs["token"] = "anon"  # noqa: S105

        return cast(GCSFileSystem, super().create_fs(**kwargs))

    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """
        Parse timestamp string returned by GCSFileSystem.

        This ensures that the passed timestamp is timezone aware.
        """
        dt = isoparse(timestamp)
        assert dt.tzinfo is not None
        return dt

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        prefix = start_prefix
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        found = False
        try:
            page_queue: PageQueue = asyncio.Queue(2)
            consumer = asyncio.create_task(
                self._process_pages(page_queue, result_queue)
            )
            try:
                await self._get_pages(prefix, page_queue)
                found = await consumer
                if not found:
                    raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")
            finally:
                consumer.cancel()  # In case _get_pages() raised
        finally:
            result_queue.put_nowait(None)

    _fetch_default = _fetch_flat

    async def _process_pages(
        self, page_queue: PageQueue, result_queue: ResultQueue
    ) -> bool:
        found = False
        with tqdm(desc=f"Listing {self.uri}", unit=" objects") as pbar:
            while (page := await page_queue.get()) is not None:
                if page:
                    found = True
                entries = [
                    self._entry_from_dict(d)
                    for d in page
                    if self._is_valid_key(d["name"])
                ]
                if entries:
                    await result_queue.put(entries)
                    pbar.update(len(entries))
        return found

    async def _get_pages(self, path: str, page_queue: PageQueue) -> None:
        page_size = 5000
        try:
            next_page_token = None
            while True:
                page = await self.fs._call(
                    "GET",
                    "b/{}/o",
                    self.name,
                    delimiter="",
                    prefix=path,
                    maxResults=page_size,
                    pageToken=next_page_token,
                    json_out=True,
                    versions="true",
                )
                assert page["kind"] == "storage#objects"
                await page_queue.put(page.get("items", []))
                next_page_token = page.get("nextPageToken")
                if next_page_token is None:
                    break
        finally:
            await page_queue.put(None)

    def _entry_from_dict(self, d: dict[str, Any]) -> Entry:
        info = self.fs._process_object(self.name, d)
        full_path = info["name"]
        subprefix = self.rel_path(full_path)
        parent = posixpath.dirname(subprefix)
        return self.convert_info(info, parent)

    def convert_info(self, v: dict[str, Any], parent: str) -> Entry:
        name = v.get("name", "").split(DELIMITER)[-1]
        if "generation" in v:
            gen = f"#{v['generation']}"
            if name.endswith(gen):
                name = name[: -len(gen)]
        return Entry.from_file(
            parent=parent,
            name=name,
            etag=v.get("etag", ""),
            version=v.get("generation", ""),
            is_latest=not v.get("timeDeleted"),
            last_modified=self.parse_timestamp(v["updated"]),
            size=v.get("size", ""),
        )
