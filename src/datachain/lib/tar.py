import hashlib
import tarfile
from collections.abc import Iterator

from datachain.lib.file import File, TarVFile


def build_tar_member(
    parent: File, info: tarfile.TarInfo, file_cls: type[TarVFile] = TarVFile
) -> TarVFile:
    new_parent = parent.get_full_name()
    etag_string = "-".join([parent.etag, info.path, str(info.mtime)])
    etag = hashlib.md5(etag_string.encode(), usedforsecurity=False).hexdigest()
    return file_cls(
        source=parent.source,
        path=f"{new_parent}/{info.path}",
        version=parent.version,
        size=info.size,
        etag=etag,
        file=parent,
    )


def process_tar(file: File) -> Iterator[TarVFile]:
    with file.open() as fd:
        with tarfile.open(fileobj=fd) as tar:
            for entry in tar.getmembers():
                yield build_tar_member(file, entry)
