import hashlib
import json
import tarfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.file import File, TarVFile
from datachain.lib.utils import DataChainError


class WDSError(DataChainError):
    def __init__(self, tar_stream, message: str):
        super().__init__(f"WebDataset error '{tar_stream.get_full_name()}': {message}")


class CoreFileDuplicationError(WDSError):
    def __init__(self, tar_stream, file1: str, file2: str):
        super().__init__(
            tar_stream, f"duplication of files with core extensions: {file1}, {file2}"
        )


class CoreFileNotFoundError(WDSError):
    def __init__(self, tar_stream, extensions, stem):
        super().__init__(
            tar_stream,
            f"no files with the extensions '{','.join(extensions)}'"
            f" were found for file stem {stem}",
        )


class UnknownFileExtensionError(WDSError):
    def __init__(self, tar_stream, name, ext):
        super().__init__(tar_stream, f"unknown extension '{ext}' for file '{name}'")


class WDSBasic(DataModel):
    file: File


class WDSAllFile(WDSBasic):
    txt: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    cap: Optional[str] = Field(default=None)
    transcript: Optional[str] = Field(default=None)
    cls: Optional[int] = Field(default=None)
    cls2: Optional[int] = Field(default=None)
    index: Optional[int] = Field(default=None)
    inx: Optional[int] = Field(default=None)
    id: Optional[int] = Field(default=None)
    json: Optional[dict] = Field(default=None)  # type: ignore[assignment]
    jsn: Optional[dict] = Field(default=None)

    pyd: Optional[bytes] = Field(default=None)
    pickle: Optional[bytes] = Field(default=None)
    pth: Optional[bytes] = Field(default=None)
    ten: Optional[bytes] = Field(default=None)
    tb: Optional[bytes] = Field(default=None)
    mp: Optional[bytes] = Field(default=None)
    msg: Optional[bytes] = Field(default=None)
    npy: Optional[bytes] = Field(default=None)
    npz: Optional[bytes] = Field(default=None)
    cbor: Optional[bytes] = Field(default=None)


class WDSReadableSubclass(DataModel):
    @staticmethod
    def _reader(builder, item: tarfile.TarInfo) -> "WDSReadableSubclass":
        raise NotImplementedError


class BuilderState:
    def __init__(self):
        self.stem = None
        self.core_file = None
        self.data = {}


class Builder:
    DEFAULT_TYPES_READERS: ClassVar[dict[type, Any]] = {
        str: lambda bld, item: bld.read_text(item),
        int: lambda bld, item: int(bld.read_text(item)),
        float: lambda bld, item: float(bld.read_text(item)),
        bytes: lambda bld, item: bld.read(item),
        dict: lambda bld, item: json.loads(bld.read_text(item)),
    }

    def __init__(
        self,
        tar_stream: File,
        core_extensions: list[str],
        wds_class: type[WDSBasic],
        tar,
        encoding="utf-8",
    ):
        self._core_extensions = core_extensions
        self._tar_stream = tar_stream
        self._wds_class = wds_class
        self._tar = tar
        self._encoding = encoding
        self.state = BuilderState()

    def read(self, item):
        return self._tar.extractfile(item).read()

    def read_text(self, item):
        return self._tar.extractfile(item).read().decode(self._encoding)

    def add(self, file: tarfile.TarInfo):
        fstream = File(path=file.name)
        ext = fstream.get_file_ext()
        stem = fstream.get_file_stem()

        if self.state.stem is not None and self.state.stem != stem:
            raise StopIteration

        if self.state.stem is None:
            self.state.stem = stem

        if ext in self._core_extensions:
            if self.state.core_file is not None:
                raise CoreFileDuplicationError(
                    self._tar_stream, file.name, self.state.core_file.name
                )
            self.state.core_file = file
        elif ext in self.state.data:
            raise WDSError(
                self._tar_stream,
                f"file with extension '.{ext}' already exists in the archive",
            )
        else:
            type_ = self._get_type(ext)
            if type_ is None:
                raise UnknownFileExtensionError(self._tar_stream, fstream.name, ext)

            if issubclass(type_, WDSReadableSubclass):
                reader = type_._reader
            else:
                reader = self.DEFAULT_TYPES_READERS.get(type_, None)

            if reader is None:
                raise WDSError(
                    self._tar_stream,
                    f"unable to find a reader for type {type_}, extension .{ext}",
                )
            self.state.data[ext] = reader(self, file)

    def produce(self):
        if self.state.core_file is None:
            raise CoreFileNotFoundError(
                self._tar_stream, self._core_extensions, self.state.stem
            )

        file = self.build_file_record()
        wds = self._wds_class(**self.state.data | {"file": file})
        self.state = BuilderState()
        return wds

    def build_file_record(self):
        new_parent = self._tar_stream.get_full_name()
        core_file = self.state.core_file
        etag_string = "-".join(
            [self._tar_stream.etag, core_file.name, str(core_file.mtime)]
        )
        etag = hashlib.md5(etag_string.encode(), usedforsecurity=False).hexdigest()
        return File(
            source=self._tar_stream.source,
            path=f"{new_parent}/{core_file.name}",
            version=self._tar_stream.version,
            size=core_file.size,
            etag=etag,
            location=[
                {
                    "vtype": TarVFile.get_vtype(),
                    "parent": self._tar_stream.model_dump_custom(),
                    "size": core_file.size,
                    "offset": core_file.offset_data,
                }
            ],
        )

    def _get_type(self, ext):
        field = self._wds_class.model_fields.get(ext, None)
        if field is None:
            return

        anno = field.annotation
        if get_origin(anno) == Union:
            args = get_args(anno)
            anno = args[0]

        return anno


class TarStream(File):
    @staticmethod
    def to_text(data):
        return data.decode("utf-8")

    _DATA_CONVERTERS: ClassVar[dict[type, Any]] = {
        str: lambda data: TarStream.to_text(data),
        int: lambda data: int(TarStream.to_text(data)),
        float: lambda data: float(TarStream.to_text(data)),
        bytes: lambda data: data,
        dict: lambda data: json.loads(TarStream.to_text(data)),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tar = None

    def open(self):
        self._tar = tarfile.open(fileobj=super().open())  # noqa: SIM115
        return self

    def getmembers(self) -> list[tarfile.TarInfo]:
        return self._tar.getmembers()

    def read_member(self, member: tarfile.TarInfo, type):
        fd = self._tar.extractfile(member)
        data = fd.read()
        converter = self._DATA_CONVERTERS.get(type, None)
        if not converter:
            raise ValueError("")
        return converter(data)


def get_tar_groups(stream, tar, core_extensions, spec, encoding="utf-8"):
    builder = Builder(stream, core_extensions, spec, tar, encoding)

    for item in sorted(tar.getmembers(), key=lambda m: Path(m.name).stem):
        if not item.isfile():
            continue
        try:
            builder.add(item)
        except StopIteration:
            yield builder.produce()
            builder.add(item)
    if builder.state.stem is not None:
        yield builder.produce()


def process_webdataset(
    core_extensions: Sequence[str] = ("jpg", "png"), spec=WDSAllFile, encoding="utf-8"
) -> Callable:
    def wds_func(file: File) -> Iterator[spec]:
        with file.open() as fd:
            with tarfile.open(fileobj=fd) as tar:
                yield from get_tar_groups(file, tar, core_extensions, spec, encoding)

    return wds_func
