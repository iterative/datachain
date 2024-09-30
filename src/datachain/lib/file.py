import hashlib
import io
import json
import logging
import os
import posixpath
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from PIL import Image
from pyarrow.dataset import dataset
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from typing_extensions import Self

from datachain.lib.data_model import DataModel
from datachain.lib.utils import DataChainError
from datachain.sql.types import Boolean, DateTime, Int, String
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from typing_extensions import Self

    from datachain.catalog import Catalog
    from datachain.client.fsspec import Client
    from datachain.dataset import RowDict

sha256 = partial(hashlib.sha256, usedforsecurity=False)

logger = logging.getLogger("datachain")

# how to create file path when exporting
ExportPlacement = Literal["filename", "etag", "fullpath", "checksum"]


class FileError(DataChainError):
    def __init__(self, file: "File", message: str):
        super().__init__(f"Error in file {file.get_uri()}: {message}")


class File(DataModel):
    """`DataModel` for reading binary files."""

    source: str = Field(default="")
    path: str
    size: int = Field(default=0)
    version: str = Field(default="")
    etag: str = Field(default="")
    is_latest: bool = Field(default=True)
    last_modified: datetime = Field(default=TIME_ZERO)

    _datachain_column_types: ClassVar[dict[str, Any]] = {
        "source": String,
        "path": String,
        "size": Int,
        "version": String,
        "etag": String,
        "is_latest": Boolean,
        "last_modified": DateTime,
    }

    _unique_id_keys: ClassVar[list[str]] = [
        "source",
        "path",
        "size",
        "etag",
        "version",
        "is_latest",
        "last_modified",
    ]

    @staticmethod
    def _validate_dict(
        v: Optional[Union[str, dict, list[dict]]],
    ) -> Optional[Union[str, dict, list[dict]]]:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Unable to convert string '{v}' to dict for File feature: {e}"
                ) from None
        return v

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, path):
        return Path(path).as_posix()

    def model_dump_custom(self):
        res = self.model_dump()
        res["last_modified"] = str(res["last_modified"])
        return res

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._catalog = None
        self._caching_enabled: bool = False

    @classmethod
    def _from_row(cls, row: "RowDict") -> "Self":
        return cls(**{key: row[key] for key in cls._datachain_column_types})

    @property
    def name(self):
        return PurePosixPath(self.path).name

    @property
    def parent(self):
        return str(PurePosixPath(self.path).parent)

    @contextmanager
    def open(self, mode: Literal["rb", "r"] = "rb") -> Iterator[Any]:
        """Open the file and return a file object."""
        if self._caching_enabled:
            self.ensure_cached()
        client: Client = self._catalog.get_client(self.source)
        with client.open_object(
            self, use_cache=self._caching_enabled, cb=self._download_cb
        ) as f:
            yield io.TextIOWrapper(f) if mode == "r" else f

    def read(self, length: int = -1):
        """Returns file contents."""
        with self.open() as stream:
            return stream.read(length)

    def read_bytes(self):
        """Returns file contents as bytes."""
        return self.read()

    def read_text(self):
        """Returns file contents as text."""
        with self.open(mode="r") as stream:
            return stream.read()

    def save(self, destination: str):
        """Writes it's content to destination"""
        with open(destination, mode="wb") as f:
            f.write(self.read())

    def export(
        self,
        output: str,
        placement: ExportPlacement = "fullpath",
        use_cache: bool = True,
    ) -> None:
        """Export file to new location."""
        if use_cache:
            self._caching_enabled = use_cache
        dst = self.get_destination_path(output, placement)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        self.save(dst)

    def _set_stream(
        self,
        catalog: "Catalog",
        caching_enabled: bool = False,
        download_cb: Callback = DEFAULT_CALLBACK,
    ) -> None:
        self._catalog = catalog
        self._caching_enabled = caching_enabled
        self._download_cb = download_cb

    def ensure_cached(self) -> None:
        if self._catalog is None:
            raise RuntimeError(
                "cannot download file to cache because catalog is not setup"
            )
        client = self._catalog.get_client(self.source)
        client.download(self, callback=self._download_cb)

    def get_local_path(self) -> Optional[str]:
        """Return path to a file in a local cache.

        Returns None if file is not cached.
        Raises an exception if cache is not setup.
        """
        if self._catalog is None:
            raise RuntimeError(
                "cannot resolve local file path because catalog is not setup"
            )
        return self._catalog.cache.get_path(self)

    def get_file_suffix(self):
        """Returns last part of file name with `.`."""
        return PurePosixPath(self.path).suffix

    def get_file_ext(self):
        """Returns last part of file name without `.`."""
        return PurePosixPath(self.path).suffix.strip(".")

    def get_file_stem(self):
        """Returns file name without extension."""
        return PurePosixPath(self.path).stem

    def get_full_name(self):
        """Returns name with parent directories."""
        return self.path

    def get_uri(self):
        """Returns file URI."""
        return f"{self.source}/{self.get_full_name()}"

    def get_path(self) -> str:
        """Returns file path."""
        path = unquote(self.get_uri())
        source = urlparse(self.source)
        if source.scheme == "file":
            path = urlparse(path).path
            path = url2pathname(path)
        return path

    def get_destination_path(self, output: str, placement: ExportPlacement) -> str:
        """
        Returns full destination path of a file for exporting to some output
        based on export placement
        """
        if placement == "filename":
            path = unquote(self.name)
        elif placement == "etag":
            path = f"{self.etag}{self.get_file_suffix()}"
        elif placement == "fullpath":
            path = unquote(self.get_full_name())
            source = urlparse(self.source)
            if source.scheme and source.scheme != "file":
                path = posixpath.join(source.netloc, path)
        elif placement == "checksum":
            raise NotImplementedError("Checksum placement not implemented yet")
        else:
            raise ValueError(f"Unsupported file export placement: {placement}")
        return posixpath.join(output, path)  # type: ignore[union-attr]

    def get_fs(self):
        """Returns `fsspec` filesystem for the file."""
        return self._catalog.get_client(self.source).fs

    def get_hash(self) -> str:
        fingerprint = f"{self.source}/{self.path}/{self.version}/{self.etag}"
        return sha256(fingerprint.encode()).hexdigest()

    def resolve(self) -> "Self":
        """
        Resolve a File object by checking its existence and updating its metadata.

        Returns:
            File: The resolved File object with updated metadata.
        """
        if self._catalog is None:
            raise RuntimeError("Cannot resolve file: catalog is not set")

        try:
            client = self._catalog.get_client(self.source)
        except NotImplementedError as e:
            raise RuntimeError(
                f"Unsupported protocol for file source: {self.source}"
            ) from e

        try:
            info = client.fs.info(client.get_full_path(self.path))
            converted_info = client.info_to_file(info, self.source)
            return type(self)(
                path=self.path,
                source=self.source,
                size=converted_info.size,
                etag=converted_info.etag,
                version=converted_info.version,
                is_latest=converted_info.is_latest,
                last_modified=converted_info.last_modified,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.warning("File system error when resolving %s: %s", self.path, str(e))

        return type(self)(
            path=self.path,
            source=self.source,
            size=0,
            etag="",
            version="",
            is_latest=True,
            last_modified=TIME_ZERO,
        )


def resolve(file: File) -> File:
    """
    Resolve a File object by checking its existence and updating its metadata.

    This function is a wrapper around the File.resolve() method, designed to be
    used as a mapper in DataChain operations.

    Args:
        file (File): The File object to resolve.

    Returns:
        File: The resolved File object with updated metadata.

    Raises:
        RuntimeError: If the file's catalog is not set or if
        the file source protocol is unsupported.
    """
    return file.resolve()


class TextFile(File):
    """`DataModel` for reading text files."""

    @contextmanager
    def open(self, mode: Literal["rb", "r"] = "r"):
        """Open the file and return a file object (default to text mode)."""
        with super().open(mode=mode) as stream:
            yield stream

    def read_text(self):
        """Returns file contents as text."""
        with self.open() as stream:
            return stream.read()

    def save(self, destination: str):
        """Writes it's content to destination"""
        with open(destination, mode="w") as f:
            f.write(self.read_text())


class ImageFile(File):
    """`DataModel` for reading image files."""

    def read(self):
        """Returns `PIL.Image.Image` object."""
        with self.open() as stream:
            return Image.open(BytesIO(stream.read()))

    def save(self, destination: str):
        """Writes it's content to destination"""
        self.read().save(destination)


class TarVFile(File):
    """`DataModel` for files extracted from tar archives."""

    tar: File

    @contextmanager
    def open(self):
        """Stream file from tar archive."""
        vpath = str(PurePosixPath(self.path).relative_to(self.tar.path))
        with self.tar.open() as fd:
            # See https://stackoverflow.com/a/32476914/3127500
            with tarfile.open(fileobj=fd, mode="r|*") as tar:
                for vfile in tar:
                    if vfile.path == vpath:
                        yield tar.extractfile(vfile)
                        break


class ArrowRow(DataModel):
    """`DataModel` for reading row from Arrow-supported file."""

    file: File
    index: int
    kwargs: dict

    @contextmanager
    def open(self):
        """Stream row contents from indexed file."""
        if self.file._caching_enabled:
            self.file.ensure_cached()
            path = self.file.get_local_path()
            ds = dataset(path, **self.kwargs)

        else:
            path = self.file.get_path()
            ds = dataset(path, filesystem=self.file.get_fs(), **self.kwargs)

        return ds.take([self.index]).to_reader()

    def read(self):
        """Returns row contents as dict."""
        with self.open() as record_batch:
            return record_batch.to_pylist()[0]


def get_file_type(type_: Literal["binary", "text", "image"] = "binary") -> type[File]:
    file: type[File] = File
    if type_ == "text":
        file = TextFile
    elif type_ == "image":
        file = ImageFile  # type: ignore[assignment]

    return file
