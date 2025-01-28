import errno
import hashlib
import io
import json
import logging
import os
import posixpath
from abc import ABC, abstractmethod
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
from PIL import Image as PilImage
from pydantic import Field, field_validator

from datachain.client.fileslice import FileSlice
from datachain.lib.data_model import DataModel
from datachain.lib.utils import DataChainError
from datachain.sql.types import JSON, Boolean, DateTime, Int, String
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from numpy import ndarray
    from typing_extensions import Self

    from datachain.catalog import Catalog
    from datachain.client.fsspec import Client
    from datachain.dataset import RowDict

sha256 = partial(hashlib.sha256, usedforsecurity=False)

logger = logging.getLogger("datachain")

# how to create file path when exporting
ExportPlacement = Literal["filename", "etag", "fullpath", "checksum"]

FileType = Literal["binary", "text", "image", "video"]


class VFileError(DataChainError):
    def __init__(self, file: "File", message: str, vtype: str = ""):
        type_ = f" of vtype '{vtype}'" if vtype else ""
        super().__init__(f"Error in v-file '{file.path}'{type_}: {message}")


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
            raise VFileError(file, "multiple 'location's are not supported yet")

        loc = location[0]

        if (offset := loc.get("offset", None)) is None:
            raise VFileError(file, "'offset' is not specified")

        if (size := loc.get("size", None)) is None:
            raise VFileError(file, "'size' is not specified")

        if (parent := loc.get("parent", None)) is None:
            raise VFileError(file, "'parent' is not specified")

        tar_file = File(**parent)
        tar_file._set_stream(file._catalog)

        client = file._catalog.get_client(tar_file.source)
        fd = client.open_object(tar_file, use_cache=file._caching_enabled)
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
        self._caching_enabled: bool = False

    @classmethod
    def upload(
        cls, data: bytes, path: str, catalog: Optional["Catalog"] = None
    ) -> "File":
        parent, name = posixpath.split(path)
        catalog, client = get_client_from_path(parent, catalog=catalog)
        file = client.upload(data, name)
        file._set_stream(catalog)
        return file

    @classmethod
    def from_local_path(cls, path: str, catalog: Optional["Catalog"] = None) -> "File":
        parent, name = posixpath.split(path)
        catalog, client = get_client_from_path(parent, catalog=catalog)
        file_info = client.fs.info(path)
        file = client.info_to_file(file_info, name)
        file._set_stream(catalog)
        return file

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
        if self.location:
            with VFileRegistry.resolve(self, self.location) as f:  # type: ignore[arg-type]
                yield f

        else:
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

    def _symlink_to(self, destination: str):
        if self.location:
            raise OSError(errno.ENOTSUP, "Symlinking virtual file is not supported")

        if self._caching_enabled:
            self.ensure_cached()
            source = self.get_local_path()
            assert source, "File was not cached"
        elif self.source.startswith("file://"):
            source = self.get_path()
        else:
            raise OSError(errno.EXDEV, "can't link across filesystems")
        return os.symlink(source, destination)

    def export(
        self,
        output: str,
        placement: ExportPlacement = "fullpath",
        use_cache: bool = True,
        link_type: Literal["copy", "symlink"] = "copy",
    ) -> None:
        """Export file to new location."""
        if use_cache:
            self._caching_enabled = use_cache
        dst = self.get_destination_path(output, placement)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        if link_type == "symlink":
            try:
                return self._symlink_to(dst)
            except OSError as exc:
                if exc.errno not in (errno.ENOTSUP, errno.EXDEV, errno.ENOSYS):
                    raise

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

    async def _prefetch(self, download_cb: Optional["Callback"] = None) -> bool:
        from datachain.client.hf import HfClient

        if self._catalog is None:
            raise RuntimeError("cannot prefetch file because catalog is not setup")

        client = self._catalog.get_client(self.source)
        if client.protocol == HfClient.protocol:
            return False

        await client._download(self, callback=download_cb or self._download_cb)
        self._set_stream(
            self._catalog, caching_enabled=True, download_cb=DEFAULT_CALLBACK
        )
        return True

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
        if self.location:
            fingerprint += f"/{self.location}"
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
            converted_info = client.info_to_file(info, self.path)
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
        return PilImage.open(BytesIO(fobj))

    def save(self, destination: str):
        """Writes it's content to destination"""
        self.read().save(destination)


class Image(DataModel):
    """`DataModel` for image file meta information."""

    width: int = Field(default=-1)
    height: int = Field(default=-1)
    format: str = Field(default="")


class VideoFile(File):
    """`DataModel` for reading video files."""

    def get_info(self) -> "Video":
        """Returns video file information."""
        from .video import video_info

        return video_info(self)

    def get_frame_np(self, frame: int) -> "ndarray":
        """
        Reads video frame from a file.

        Args:
            frame (int): Frame number to read.

        Returns:
            ndarray: Video frame.
        """
        from .video import video_frame_np

        return video_frame_np(self, frame)

    def get_frame(self, frame: int, format: str = "jpg") -> bytes:
        """
        Reads video frame from a file and returns as image bytes.

        Args:
            frame (int): Frame number to read.
            format (str): Image format (default: 'jpg').

        Returns:
            bytes: Video frame image as bytes.
        """
        from .video import video_frame

        return video_frame(self, frame, format)

    def save_frame(self, frame: int, output_file: str) -> "VideoFrame":
        """
        Saves video frame as an image file.

        Args:
            frame (int): Frame number to read.
            output_file (str): Output file path.

        Returns:
            VideoFrame: Video frame model.
        """
        from .video import save_video_frame

        return save_video_frame(self, frame, output_file)

    def get_frames_np(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> "Iterator[ndarray]":
        """
        Reads video frames from a file.

        Args:
            start_frame (int): Frame number to start reading from (default: 0).
            end_frame (int): Frame number to stop reading at (default: None).
            step (int): Step size for reading frames (default: 1).

        Returns:
            Iterator[ndarray]: Iterator of video frames.
        """
        from .video import video_frames_np

        return video_frames_np(self, start_frame, end_frame, step)

    def get_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        format: str = "jpg",
    ) -> "Iterator[bytes]":
        """
        Reads video frames from a file and returns as bytes.

        Args:
            start_frame (int): Frame number to start reading from (default: 0).
            end_frame (int): Frame number to stop reading at (default: None).
            step (int): Step size for reading frames (default: 1).
            format (str): Image format (default: 'jpg').

        Returns:
            Iterator[bytes]: Iterator of video frames.
        """
        from .video import video_frames

        return video_frames(self, start_frame, end_frame, step, format)

    def save_frames(
        self,
        output_dir: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        format: str = "jpg",
    ) -> "Iterator[VideoFrame]":
        """
        Saves video frames as image files.

        Args:
            output_dir (str): Output directory path.
            start_frame (int): Frame number to start reading from (default: 0).
            end_frame (int): Frame number to stop reading at (default: None).
            step (int): Step size for reading frames (default: 1).
            format (str): Image format (default: 'jpg').

        Returns:
            Iterator[VideoFrame]: List of video frame models.
        """
        from .video import save_video_frames

        return save_video_frames(self, output_dir, start_frame, end_frame, step, format)

    def save_fragment(
        self,
        start_time: float,
        end_time: float,
        output_file: str,
    ) -> "VideoFragment":
        """
        Saves video interval as a new video file.

        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            output_file (str): Output file path.

        Returns:
            VideoFragment: Video fragment model.
        """
        from .video import save_video_fragment

        return save_video_fragment(self, start_time, end_time, output_file)

    def save_fragments(
        self,
        intervals: list[tuple[float, float]],
        output_dir: str,
    ) -> "Iterator[VideoFragment]":
        """
        Saves video intervals as new video files.

        Args:
            intervals (list[tuple[float, float]]): List of start and end times
                                                   in seconds.
            output_dir (str): Output directory path.

        Returns:
            Iterator[VideoFragment]: List of video fragment models.
        """
        from .video import save_video_fragments

        return save_video_fragments(self, intervals, output_dir)


class VideoFragment(VideoFile):
    """`DataModel` for reading video fragments."""

    start: float = Field(default=-1.0)
    end: float = Field(default=-1.0)
    orig: File


class VideoFrame(VideoFile):
    """`DataModel` for reading video frames."""

    frame: int = Field(default=-1)
    timestamp: float = Field(default=-1.0)
    orig: File


class Video(DataModel):
    """`DataModel` for video file meta information."""

    width: int = Field(default=-1)
    height: int = Field(default=-1)
    fps: float = Field(default=-1.0)
    duration: float = Field(default=-1.0)
    frames: int = Field(default=-1)
    format: str = Field(default="")
    codec: str = Field(default="")


class Frame(DataModel):
    """`DataModel` for video frame image meta information."""

    frame: int = Field(default=-1)
    timestamp: float = Field(default=-1.0)
    width: int = Field(default=-1)
    height: int = Field(default=-1)
    format: str = Field(default="")


class ArrowRow(DataModel):
    """`DataModel` for reading row from Arrow-supported file."""

    file: File
    index: int
    kwargs: dict

    @contextmanager
    def open(self):
        """Stream row contents from indexed file."""
        from pyarrow.dataset import dataset

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


def get_file_type(type_: FileType = "binary") -> type[File]:
    file: type[File] = File
    if type_ == "text":
        file = TextFile
    elif type_ == "image":
        file = ImageFile  # type: ignore[assignment]
    elif type_ == "video":
        file = VideoFile

    return file


def get_client_from_path(
    path: str, catalog: Optional["Catalog"] = None
) -> tuple["Catalog", "Client"]:
    if catalog is None:
        from datachain.catalog.loader import get_catalog

        catalog = get_catalog()

    return catalog, catalog.get_client(path)
