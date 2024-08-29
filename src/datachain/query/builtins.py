import hashlib
import tarfile
from functools import partial

from datachain.lib.file import File
from datachain.sql.types import JSON, Boolean, DateTime, Int64, String

from .schema import C, Object
from .udf import udf

md5 = partial(hashlib.md5, usedforsecurity=False)

__all__ = ["checksum", "index_tar"]


def load_tar(raw):
    with tarfile.open(fileobj=raw, mode="r:") as tar:
        return tar.getmembers()


@udf(
    (
        C("file.source"),
        C("file.path"),
        C("file.size"),
        C("file.version"),
        C("file.etag"),
        C("file.is_latest"),
        C("file.last_modified"),
        C("file.vtype"),
        Object(load_tar),
    ),
    {
        "file__source": String,
        "file__path": String,
        "file__size": Int64,
        "file__version": String,
        "file__etag": String,
        "file__is_latest": Boolean,
        "file__last_modified": DateTime,
        "file__location": JSON,
        "file__vtype": String,
    },
)
def index_tar(
    source,
    parent_path,
    size,
    version,
    etag,
    is_latest,
    last_modified,
    vtype,
    tar_entries,
):
    # generate original tar files as well, along with subobjects
    file = File(
        source=source,
        path=parent_path,
        size=size,
        version=version,
        etag=etag,
        is_latest=is_latest,
        last_modified=last_modified,
        vtype=vtype,
    )
    yield tuple(f[1] for f in list(file))

    for info in tar_entries:
        if info.isfile():
            full_path = f"{parent_path}/{info.name}"
            file = File(
                source=source,
                path=full_path,
                size=info.size,
                vtype="tar",
                location=[
                    {
                        "vtype": "tar",
                        "offset": info.offset_data,
                        "size": info.size,
                        "parent": {
                            "source": source,
                            "path": parent_path,
                            "version": version,
                            "size": size,
                            "etag": etag,
                            "vtype": "",
                            "location": None,
                        },
                    }
                ],
            )
            yield tuple(f[1] for f in list(file))


BUFSIZE = 2**18


def file_digest(fileobj):
    """Calculate the digest of a file-like object."""
    buf = bytearray(BUFSIZE)  # Reusable buffer to reduce allocations.
    view = memoryview(buf)
    digestobj = md5()
    # From 3.11's hashlib.filedigest()
    while True:
        size = fileobj.readinto(buf)
        if size == 0:
            break  # EOF
        digestobj.update(view[:size])
    return digestobj.hexdigest()


@udf(params=[Object(file_digest)], output={"checksum": String})
def checksum(digest):
    return (digest,)
