import io
import json
import logging
import os
import posixpath
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from PIL import Image
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from typing_extensions import Self

from datachain.cache import UniqueId
from datachain.client.fileslice import FileSlice
from datachain.lib.data_model import DataModel
from datachain.lib.utils import DataChainError
from datachain.sql.types import JSON, Boolean, DateTime, Int, String
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from datachain.catalog import Catalog

logger = logging.getLogger("datachain")

# how to create file path when exporting
ExportPlacement = Literal["filename", "etag", "fullpath", "checksum"]


class VFileError(DataChainError):
    def __init__(self, file: "File", message: str, vtype: str = ""):
        type_ = f" of vtype '{vtype}'" if vtype else ""
        super().__init__(f"Error in v-file '{file.get_uid().path}'{type_}: {message}")


class FileError(DataChainError):
    def __init__(self, file: "File", message: str):
        super().__init__(f"Error in file {file.get_uri()}: {message}")


class VFile(ABC):
    @classmethod
    @abstractmethod
    def get_vtype(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def open(cls, file: "File", location: list[dict]):
        pass


class TarVFile(VFile):
    """Virtual file model for files extracted from tar archives."""

    @classmethod
    def get_vtype(cls) -> str:
        return "tar"

    @classmethod
    def open(cls, file: "File", location: list[dict]):
        """Stream file from tar archive based on location in archive."""
        if len(location) > 1:
            VFileError(file, "multiple 'location's are not supported yet")

        loc = location[0]

        if (offset := loc.get("offset", None)) is None:
            VFileError(file, "'offset' is not specified")

        if (size := loc.get("size", None)) is None:
            VFileError(file, "'size' is not specified")

        if (parent := loc.get("parent", None)) is None:
            VFileError(file, "'parent' is not specified")

        tar_file = File(**parent)
        tar_file._set_stream(file._catalog)

        tar_file_uid = tar_file.get_uid()
        client = file._catalog.get_client(tar_file_uid.storage)
        fd = client.open_object(tar_file_uid, use_cache=file._caching_enabled)
        return FileSlice(fd, offset, size, file.name)


class VFileRegistry:
    _vtype_readers: ClassVar[dict[str, type["VFile"]]] = {"tar": TarVFile}

    @classmethod
    def register(cls, reader: type["VFile"]):
        cls._vtype_readers[reader.get_vtype()] = reader

    @classmethod
    def resolve(cls, file: "File", location: list[dict]):
        if len(location) == 0:
            raise VFileError(file, "'location' must not be list of JSONs")

        if not (vtype := location[0].get("vtype", "")):
            raise VFileError(file, "vtype is not specified")

        reader = cls._vtype_readers.get(vtype, None)
        if not reader:
            raise VFileError(file, "reader not registered", vtype)

        return reader.open(file, location)


class File(DataModel):
    """`DataModel` for reading binary files."""

    source: str = Field(default="")
    path: str
    size: int = Field(default=0)
    version: str = Field(default="")
    etag: str = Field(default="")
    is_latest: bool = Field(default=True)
    last_modified: datetime = Field(default=TIME_ZERO)
    location: Optional[Union[dict, list[dict]]] = Field(default=None)

    _datachain_column_types: ClassVar[dict[str, Any]] = {
        "source": String,
        "path": String,
        "size": Int,
        "version": String,
        "etag": String,
        "is_latest": Boolean,
        "last_modified": DateTime,
        "location": JSON,
    }

    _unique_id_keys: ClassVar[list[str]] = [
        "source",
        "path",
        "size",
        "etag",
        "version",
        "is_latest",
        "location",
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

    # Workaround for empty JSONs converted to empty strings in some DBs.
    @field_validator("location", mode="before")
    @classmethod
    def validate_location(cls, v):
        return File._validate_dict(v)

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
        self._caching_enabled = False

    @property
    def name(self):
        return PurePosixPath(self.path).name

    @property
    def parent(self):
        return str(PurePosixPath(self.path).parent)

    @contextmanager
    def open(self, mode: Literal["rb", "r"] = "rb"):
        """Open the file and return a file object."""
        if self.location:
            with VFileRegistry.resolve(self, self.location) as f:  # type: ignore[arg-type]
                yield f

        else:
            uid = self.get_uid()
            client = self._catalog.get_client(self.source)
            if self._caching_enabled:
                client.download(uid, callback=self._download_cb)
            with client.open_object(
                uid, use_cache=self._caching_enabled, cb=self._download_cb
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

    def get_uid(self) -> UniqueId:
        """Returns unique ID for file."""
        dump = self.model_dump()
        return UniqueId(*(dump[k] for k in self._unique_id_keys))

    def get_local_path(self, download: bool = False) -> Optional[str]:
        """Returns path to a file in a local cache.
        Return None if file is not cached. Throws an exception if cache is not setup."""
        if self._catalog is None:
            raise RuntimeError(
                "cannot resolve local file path because catalog is not setup"
            )
        uid = self.get_uid()
        if download:
            client = self._catalog.get_client(self.source)
            client.download(uid, callback=self._download_cb)
        return self._catalog.cache.get_path(uid)

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
                location=self.location,
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
            location=self.location,
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
        fobj = super().read()
        return Image.open(BytesIO(fobj))

    def save(self, destination: str):
        """Writes it's content to destination"""
        self.read().save(destination)


class IndexedFile(DataModel):
    """Metadata indexed from tabular files.

    Includes `file` and `index` signals.
    """

    file: File
    index: int


def get_file_type(type_: Literal["binary", "text", "image"] = "binary") -> type[File]:
    file: type[File] = File
    if type_ == "text":
        file = TextFile
    elif type_ == "image":
        file = ImageFile  # type: ignore[assignment]

    return file
