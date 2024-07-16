import io
import json
import os
import posixpath
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional, Union
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from fsspec.implementations.local import LocalFileSystem
from PIL import Image
from pydantic import Field, field_validator

from datachain.cache import UniqueId
from datachain.client.fileslice import FileSlice
from datachain.lib.data_model import DataModel, FileBasic
from datachain.lib.utils import DataChainError
from datachain.sql.types import JSON, Int, String
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from datachain.catalog import Catalog

ExportStrategy = Literal["filename", "etag", "fullpath", "checksum"]


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
    @classmethod
    def get_vtype(cls) -> str:
        return "tar"

    @classmethod
    def open(cls, file: "File", location: list[dict]):
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


class File(FileBasic):
    source: str = Field(default="")
    parent: str = Field(default="")
    name: str
    size: int = Field(default=0)
    version: str = Field(default="")
    etag: str = Field(default="")
    is_latest: bool = Field(default=True)
    last_modified: datetime = Field(default=TIME_ZERO)
    location: Optional[Union[dict, list[dict]]] = Field(default=None)
    vtype: str = Field(default="")

    _datachain_column_types: ClassVar[dict[str, Any]] = {
        "source": String,
        "parent": String,
        "name": String,
        "version": String,
        "etag": String,
        "size": Int,
        "vtype": String,
        "location": JSON,
    }

    _unique_id_keys: ClassVar[list[str]] = [
        "source",
        "parent",
        "name",
        "etag",
        "size",
        "vtype",
        "location",
    ]

    @staticmethod
    def to_dict(
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
        return File.to_dict(v)

    @field_validator("parent", mode="before")
    @classmethod
    def validate_path(cls, path):
        if path == "":
            return ""
        return Path(path).as_posix()

    def model_dump_custom(self):
        res = self.model_dump()
        res["last_modified"] = str(res["last_modified"])
        return res

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._catalog = None
        self._caching_enabled = False

    @contextmanager
    def open(self):
        if self.location:
            with VFileRegistry.resolve(self, self.location) as f:
                yield f

        uid = self.get_uid()
        client = self._catalog.get_client(self.source)
        if self._caching_enabled:
            client.download(uid, callback=self._download_cb)
        with client.open_object(
            uid, use_cache=self._caching_enabled, cb=self._download_cb
        ) as f:
            yield f

    def export(self, output: str, strategy: ExportStrategy) -> None:
        self._set_stream(self._catalog, caching_enabled=True)
        dst = self.get_destination_path(output, strategy)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        with open(dst, mode="wb") as f:
            f.write(self.read())

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
        dump = self.model_dump()
        return UniqueId(*(dump[k] for k in self._unique_id_keys))

    def get_local_path(self) -> Optional[str]:
        """Get path to a file in a local cache.
        Return None if file is not cached. Throws an exception if cache is not setup."""
        if self._catalog is None:
            raise RuntimeError(
                "cannot resolve local file path because catalog is not setup"
            )
        return self._catalog.cache.get_path(self.get_uid())

    def get_file_suffix(self):
        return Path(self.name).suffix

    def get_file_ext(self):
        return Path(self.name).suffix.strip(".")

    def get_file_stem(self):
        return Path(self.name).stem

    def get_full_name(self):
        return (Path(self.parent) / self.name).as_posix()

    def get_uri(self):
        return f"{self.source}/{self.get_full_name()}"

    def get_path(self) -> str:
        path = unquote(self.get_uri())
        fs = self.get_fs()
        if isinstance(fs, LocalFileSystem):
            # Drop file:// protocol
            path = urlparse(path).path
            path = url2pathname(path)
        return path

    def get_destination_path(self, output: str, strategy: ExportStrategy) -> str:
        """
        Returns full destination path of a file for exporting to some output
        based on export strategy
        """
        if strategy == "filename":
            path = unquote(self.name)
        elif strategy == "etag":
            path = f"{self.etag}{self.get_file_suffix()}"
        elif strategy == "fullpath":
            fs = self.get_fs()
            if isinstance(fs, LocalFileSystem):
                path = self.get_path().lstrip(os.sep)
            else:
                path = (
                    Path(urlparse(self.source).netloc) / unquote(self.get_full_name())
                ).as_posix()
        elif strategy == "checksum":
            raise NotImplementedError("Checksum strategy not implemented yet")
        else:
            raise ValueError(f"Unsupported file export strategy: {strategy}")

        return posixpath.join(output, path)  # type: ignore[union-attr]

    def get_fs(self):
        return self._catalog.get_client(self.source).fs


class TextFile(File):
    @contextmanager
    def open(self):
        with super().open() as binary:
            yield io.TextIOWrapper(binary)


class ImageFile(File):
    def get_value(self):
        value = super().get_value()
        return Image.open(BytesIO(value))


def get_file(type_: Literal["binary", "text", "image"] = "binary"):
    file: type[File] = File
    if type_ == "text":
        file = TextFile
    elif type_ == "image":
        file = ImageFile  # type: ignore[assignment]

    def get_file_type(
        source: str,
        parent: str,
        name: str,
        version: str,
        etag: str,
        size: int,
        vtype: str,
        location: Optional[Union[dict, list[dict]]],
    ) -> file:  # type: ignore[valid-type]
        return file(
            source=source,
            parent=parent,
            name=name,
            version=version,
            etag=etag,
            size=size,
            vtype=vtype,
            location=location,
        )

    return get_file_type


class IndexedFile(DataModel):
    """File source info for tables."""

    file: File
    index: int
