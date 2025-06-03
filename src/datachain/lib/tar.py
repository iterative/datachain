import hashlib
import tarfile
from collections.abc import Iterator

from datachain.lib.file import File, TarVFile


def build_tar_member(parent: File, info: tarfile.TarInfo) -> File:
    new_parent = parent.get_full_name()
    etag_string = "-".join([parent.etag, info.name, str(info.mtime)])
    etag = hashlib.md5(etag_string.encode(), usedforsecurity=False).hexdigest()
    return File(
        source=parent.source,
        path=f"{new_parent}/{info.name}",
        version=parent.version,
        size=info.size,
        etag=etag,
        location=[
            {
                "vtype": TarVFile.get_vtype(),
                "parent": parent.model_dump_custom(),
                "size": info.size,
                "offset": info.offset_data,
            }
        ],
    )


def process_tar(file: File) -> Iterator[File]:
    with file.open() as fd:
        with tarfile.open(fileobj=fd) as tar:
            for entry in tar.getmembers():
                if entry.isfile():
                    yield build_tar_member(file, entry)
