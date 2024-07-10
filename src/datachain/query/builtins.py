import hashlib
import tarfile
from functools import partial

from datachain.sql.types import String

from .schema import C, DatasetRow, Object
from .udf import udf

md5 = partial(hashlib.md5, usedforsecurity=False)

__all__ = ["checksum", "index_tar"]


def load_tar(raw):
    with tarfile.open(fileobj=raw, mode="r:") as tar:
        return tar.getmembers()


@udf(
    (
        C.source,
        C.name,
        C.parent,
        C.size,
        C.vtype,
        C.dir_type,
        C.owner_name,
        C.owner_id,
        C.is_latest,
        C.last_modified,
        C.version,
        C.etag,
        Object(load_tar),
    ),
    DatasetRow.schema,
)
def index_tar(
    source,
    name,
    parent,
    size,
    vtype,
    dir_type,
    owner_name,
    owner_id,
    is_latest,
    last_modified,
    version,
    etag,
    tar_entries,
):
    # generate original tar files as well, along with subobjects
    yield DatasetRow.create(
        name,
        source=source,
        parent=parent,
        size=size,
        vtype=vtype,
        dir_type=dir_type,
        owner_name=owner_name,
        owner_id=owner_id,
        is_latest=bool(is_latest),
        last_modified=last_modified,
        version=version,
        etag=etag,
    )

    parent_path = name if not parent else f"{parent}/{name}"
    for info in tar_entries:
        if info.isfile():
            full_path = f"{parent_path}/{info.name}"
            parent_dir, subobject_name = full_path.rsplit("/", 1)
            yield DatasetRow.create(
                subobject_name,
                source=source,
                parent=parent_dir,
                size=info.size,
                vtype="tar",
                location={
                    "vtype": "tar",
                    "offset": info.offset_data,
                    "size": info.size,
                    "parent": {
                        "source": source,
                        "parent": parent,
                        "name": name,
                        "version": version,
                        "size": size,
                        "etag": etag,
                        "vtype": "",
                        "location": None,
                    },
                },
            )


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
