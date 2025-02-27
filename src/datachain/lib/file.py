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
from fsspec.utils import stringify_path
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
    """
    `DataModel` for reading binary files.

    Attributes:
        source (str): The source of the file (e.g., 's3://bucket-name/').
        path (str): The path to the file (e.g., 'path/to/file.txt').
        size (int): The size of the file in bytes. Defaults to 0.
        version (str): The version of the file. Defaults to an empty string.
        etag (str): The ETag of the file. Defaults to an empty string.
        is_latest (bool): Whether the file is the latest version. Defaults to `True`.
        last_modified (datetime): The last modified timestamp of the file.
            Defaults to Unix epoch (`1970-01-01T00:00:00`).
        location (dict | list[dict], optional): The location of the file.
            Defaults to `None`.
    """

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
    ) -> "Self":
        if catalog is None:
            from datachain.catalog.loader import get_catalog

            catalog = get_catalog()

        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(path)
        source, rel_path = client_cls.split_url(path)

        client = catalog.get_client(client_cls.get_uri(source))
        file = client.upload(data, rel_path)
        if not isinstance(file, cls):
            file = cls(**file.model_dump())
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
        destination = stringify_path(destination)
        client: Client = self._catalog.get_client(destination)

        if client.PREFIX == "file://" and not destination.startswith(client.PREFIX):
            destination = Path(destination).absolute().as_uri()

        client.upload(self.read(), destination)

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
        client: Client = self._catalog.get_client(dst_dir)
        client.fs.makedirs(dst_dir, exist_ok=True)

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
        destination = stringify_path(destination)

        client: Client = self._catalog.get_client(destination)
        with client.fs.open(destination, mode="w") as f:
            f.write(self.read_text())


class ImageFile(File):
    """`DataModel` for reading image files."""

    def read(self):
        """Returns `PIL.Image.Image` object."""
        fobj = super().read()
        return PilImage.open(BytesIO(fobj))

    def save(self, destination: str):
        """Writes it's content to destination"""
        destination = stringify_path(destination)

        client: Client = self._catalog.get_client(destination)
        with client.fs.open(destination, mode="wb") as f:
            self.read().save(f)


class Image(DataModel):
    """
    A data model representing metadata for an image file.

    Attributes:
        width (int): The width of the image in pixels. Defaults to -1 if unknown.
        height (int): The height of the image in pixels. Defaults to -1 if unknown.
        format (str): The format of the image file (e.g., 'jpg', 'png').
                      Defaults to an empty string.
    """

    width: int = Field(default=-1)
    height: int = Field(default=-1)
    format: str = Field(default="")


class VideoFile(File):
    """
    A data model for handling video files.

    This model inherits from the `File` model and provides additional functionality
    for reading video files, extracting video frames, and splitting videos into
    fragments.
    """

    def get_info(self) -> "Video":
        """
        Retrieves metadata and information about the video file.

        Returns:
            Video: A Model containing video metadata such as duration,
                   resolution, frame rate, and codec details.
        """
        from .video import video_info

        return video_info(self)

    def get_frame(self, frame: int) -> "VideoFrame":
        """
        Returns a specific video frame by its frame number.

        Args:
            frame (int): The frame number to read.

        Returns:
            VideoFrame: Video frame model.
        """
        if frame < 0:
            raise ValueError("frame must be a non-negative integer")

        return VideoFrame(video=self, frame=frame)

    def get_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ) -> "Iterator[VideoFrame]":
        """
        Returns video frames from the specified range in the video.

        Args:
            start (int): The starting frame number (default: 0).
            end (int, optional): The ending frame number (exclusive). If None,
                                 frames are read until the end of the video
                                 (default: None).
            step (int): The interval between frames to read (default: 1).

        Returns:
            Iterator[VideoFrame]: An iterator yielding video frames.

        Note:
            If end is not specified, number of frames will be taken from the video file,
            this means video file needs to be downloaded.
        """
        from .video import validate_frame_range

        start, end, step = validate_frame_range(self, start, end, step)

        for frame in range(start, end, step):
            yield self.get_frame(frame)

    def get_fragment(self, start: float, end: float) -> "VideoFragment":
        """
        Returns a video fragment from the specified time range.

        Args:
            start (float): The start time of the fragment in seconds.
            end (float): The end time of the fragment in seconds.

        Returns:
            VideoFragment: A Model representing the video fragment.
        """
        if start < 0 or end < 0 or start >= end:
            raise ValueError(f"Invalid time range: ({start:.3f}, {end:.3f})")

        return VideoFragment(video=self, start=start, end=end)

    def get_fragments(
        self,
        duration: float,
        start: float = 0,
        end: Optional[float] = None,
    ) -> "Iterator[VideoFragment]":
        """
        Splits the video into multiple fragments of a specified duration.

        Args:
            duration (float): The duration of each video fragment in seconds.
            start (float): The starting time in seconds (default: 0).
            end (float, optional): The ending time in seconds. If None, the entire
                                   remaining video is processed (default: None).

        Returns:
            Iterator[VideoFragment]: An iterator yielding video fragments.

        Note:
            If end is not specified, number of frames will be taken from the video file,
            this means video file needs to be downloaded.
        """
        if duration <= 0:
            raise ValueError("duration must be a positive float")
        if start < 0:
            raise ValueError("start must be a non-negative float")

        if end is None:
            end = self.get_info().duration

        if end < 0:
            raise ValueError("end must be a non-negative float")
        if start >= end:
            raise ValueError("start must be less than end")

        while start < end:
            yield self.get_fragment(start, min(start + duration, end))
            start += duration


class VideoFrame(DataModel):
    """
    A data model for representing a video frame.

    This model inherits from the `VideoFile` model and adds a `frame` attribute,
    which represents a specific frame within a video file. It allows access
    to individual frames and provides functionality for reading and saving
    video frames as image files.

    Attributes:
        video (VideoFile): The video file containing the video frame.
        frame (int): The frame number referencing a specific frame in the video file.
    """

    video: VideoFile
    frame: int

    def get_np(self) -> "ndarray":
        """
        Returns a video frame from the video file as a NumPy array.

        Returns:
            ndarray: A NumPy array representing the video frame,
                     in the shape (height, width, channels).
        """
        from .video import video_frame_np

        return video_frame_np(self.video, self.frame)

    def read_bytes(self, format: str = "jpg") -> bytes:
        """
        Returns a video frame from the video file as image bytes.

        Args:
            format (str): The desired image format (e.g., 'jpg', 'png').
                          Defaults to 'jpg'.

        Returns:
            bytes: The encoded video frame as image bytes.
        """
        from .video import video_frame_bytes

        return video_frame_bytes(self.video, self.frame, format)

    def save(self, output: str, format: str = "jpg") -> "ImageFile":
        """
        Saves the current video frame as an image file.

        If `output` is a remote path, the image file will be uploaded to remote storage.

        Args:
            output (str): The destination path, which can be a local file path
                          or a remote URL.
            format (str): The image format (e.g., 'jpg', 'png'). Defaults to 'jpg'.

        Returns:
            ImageFile: A Model representing the saved image file.
        """
        from .video import save_video_frame

        return save_video_frame(self.video, self.frame, output, format)


class VideoFragment(DataModel):
    """
    A data model for representing a video fragment.

    This model inherits from the `VideoFile` model and adds `start`
    and `end` attributes, which represent a specific fragment within a video file.
    It allows access to individual fragments and provides functionality for reading
    and saving video fragments as separate video files.

    Attributes:
        video (VideoFile): The video file containing the video fragment.
        start (float): The starting time of the video fragment in seconds.
        end (float): The ending time of the video fragment in seconds.
    """

    video: VideoFile
    start: float
    end: float

    def save(self, output: str, format: Optional[str] = None) -> "VideoFile":
        """
        Saves the video fragment as a new video file.

        If `output` is a remote path, the video file will be uploaded to remote storage.

        Args:
            output (str): The destination path, which can be a local file path
                          or a remote URL.
            format (str, optional): The output video format (e.g., 'mp4', 'avi').
                                    If None, the format is inferred from the
                                    file extension.

        Returns:
            VideoFile: A Model representing the saved video file.
        """
        from .video import save_video_fragment

        return save_video_fragment(self.video, self.start, self.end, output, format)


class Video(DataModel):
    """
    A data model representing metadata for a video file.

    Attributes:
        width (int): The width of the video in pixels. Defaults to -1 if unknown.
        height (int): The height of the video in pixels. Defaults to -1 if unknown.
        fps (float): The frame rate of the video (frames per second).
                     Defaults to -1.0 if unknown.
        duration (float): The total duration of the video in seconds.
                          Defaults to -1.0 if unknown.
        frames (int): The total number of frames in the video.
                      Defaults to -1 if unknown.
        format (str): The format of the video file (e.g., 'mp4', 'avi').
                      Defaults to an empty string.
        codec (str): The codec used for encoding the video. Defaults to an empty string.
    """

    width: int = Field(default=-1)
    height: int = Field(default=-1)
    fps: float = Field(default=-1.0)
    duration: float = Field(default=-1.0)
    frames: int = Field(default=-1)
    format: str = Field(default="")
    codec: str = Field(default="")


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
