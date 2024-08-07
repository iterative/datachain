import asyncio
from typing import Any, cast

from botocore.exceptions import NoCredentialsError
from s3fs import S3FileSystem
from tqdm import tqdm

from datachain.node import Entry

from .fsspec import DELIMITER, Client, ResultQueue

UPDATE_CHUNKSIZE = 1000


class ClientS3(Client):
    FS_CLASS = S3FileSystem
    PREFIX = "s3://"
    protocol = "s3"

    @classmethod
    def create_fs(cls, **kwargs) -> S3FileSystem:
        if "aws_endpoint_url" in kwargs:
            kwargs.setdefault("client_kwargs", {}).setdefault(
                "endpoint_url", kwargs.pop("aws_endpoint_url")
            )
        if "aws_key" in kwargs:
            kwargs.setdefault("key", kwargs.pop("aws_key"))
        if "aws_secret" in kwargs:
            kwargs.setdefault("secret", kwargs.pop("aws_secret"))
        if "aws_token" in kwargs:
            kwargs.setdefault("token", kwargs.pop("aws_token"))

        # caching bucket regions to use the right one in signed urls, otherwise
        # it tries to randomly guess and creates wrong signature
        kwargs.setdefault("cache_regions", True)

        # We want to use newer v4 signature version since regions added after
        # 2014 are not going to support v2 which is the older one.
        # All regions support v4.
        kwargs.setdefault("config_kwargs", {}).setdefault("signature_version", "s3v4")

        if not kwargs.get("anon"):
            try:
                # Run an inexpensive check to see if credentials are available
                super().create_fs(**kwargs).sign("s3://bucket/object")
            except NoCredentialsError:
                kwargs["anon"] = True
            except NotImplementedError:
                pass

        return cast(S3FileSystem, super().create_fs(**kwargs))

    async def _fetch_flat(self, start_prefix: str, result_queue: ResultQueue) -> None:
        async def get_pages(it, page_queue):
            try:
                async for page in it:
                    await page_queue.put(page.get(contents_key, []))
            finally:
                await page_queue.put(None)

        async def process_pages(page_queue, result_queue):
            found = False
            with tqdm(desc=f"Listing {self.uri}", unit=" objects") as pbar:
                while (res := await page_queue.get()) is not None:
                    if res:
                        found = True
                    entries = [
                        self._entry_from_boto(d, self.name, versions)
                        for d in res
                        if self._is_valid_key(d["Key"])
                    ]
                    if entries:
                        await result_queue.put(entries)
                        pbar.update(len(entries))
            if not found:
                raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")

        try:
            prefix = start_prefix
            if prefix:
                prefix = prefix.lstrip(DELIMITER) + DELIMITER
            versions = True
            fs = self.fs
            await fs.set_session()
            s3 = await fs.get_s3(self.name)
            if versions:
                method = "list_object_versions"
                contents_key = "Versions"
            else:
                method = "list_objects_v2"
                contents_key = "Contents"
            pag = s3.get_paginator(method)
            it = pag.paginate(
                Bucket=self.name,
                Prefix=prefix,
                Delimiter="",
            )
            page_queue: asyncio.Queue[list] = asyncio.Queue(2)
            consumer = asyncio.create_task(process_pages(page_queue, result_queue))
            try:
                await get_pages(it, page_queue)
                await consumer
            finally:
                consumer.cancel()  # In case get_pages() raised
        finally:
            result_queue.put_nowait(None)

    async def _fetch_default(
        self, start_prefix: str, result_queue: ResultQueue
    ) -> None:
        await self._fetch_flat(start_prefix, result_queue)

    def _entry_from_boto(self, v, bucket, versions=False):
        return Entry.from_file(
            path=v["Key"],
            etag=v.get("ETag", "").strip('"'),
            version=ClientS3.clean_s3_version(v.get("VersionId", "")),
            is_latest=v.get("IsLatest", True),
            last_modified=v.get("LastModified", ""),
            size=v["Size"],
            owner_name=v.get("Owner", {}).get("DisplayName", ""),
            owner_id=v.get("Owner", {}).get("ID", ""),
        )

    async def _fetch_dir(
        self,
        prefix,
        pbar,
        result_queue,
    ):
        if prefix:
            prefix = prefix.lstrip(DELIMITER) + DELIMITER
        files = []
        subdirs = set()
        found = False
        async for info in self.fs._iterdir(self.name, prefix=prefix, versions=True):
            full_path = info["name"]
            _, subprefix, _ = self.fs.split_path(full_path)
            if prefix.strip(DELIMITER) == subprefix.strip(DELIMITER):
                found = True
                continue
            if info["type"] == "directory":
                subdirs.add(subprefix)
            else:
                files.append(self.convert_info(info, subprefix))
                pbar.update()
            found = True
        if not found:
            raise FileNotFoundError(f"Unable to resolve remote path: {prefix}")
        if files:
            await result_queue.put(files)
        pbar.update(len(subdirs))
        return subdirs

    @staticmethod
    def clean_s3_version(ver):
        return ver if ver != "null" else ""

    def convert_info(self, v: dict[str, Any], path: str) -> Entry:
        return Entry.from_file(
            path=path,
            etag=v.get("ETag", "").strip('"'),
            version=ClientS3.clean_s3_version(v.get("VersionId", "")),
            is_latest=v.get("IsLatest", True),
            last_modified=v.get("LastModified", ""),
            size=v["size"],
            owner_name=v.get("Owner", {}).get("DisplayName", ""),
            owner_id=v.get("Owner", {}).get("ID", ""),
        )
