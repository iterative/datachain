from typing import TYPE_CHECKING

from fsspec.implementations.local import LocalFileSystem

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


def _isdir(fs: "AbstractFileSystem", path: str) -> bool:
    info = fs.info(path)
    return info["type"] == "directory" or (
        info["size"] == 0 and info["type"] == "file" and info["name"].endswith("/")
    )


def isfile(fs: "AbstractFileSystem", path: str) -> bool:
    """
    Returns True if uri points to a file.

    Supports special directories on object storages, e.g.:
    Google creates a zero byte file with the same name as the directory with a trailing
    slash at the end.
    """
    if isinstance(fs, LocalFileSystem):
        return fs.isfile(path)

    try:
        return not _isdir(fs, path)
    except FileNotFoundError:
        return False
