import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

from datachain.error import DataChainError


def validate_upload_args(
    source_path: str, recursive: bool, local_fs: "AbstractFileSystem"
):
    """Validate upload arguments and raise appropriate errors."""
    is_dir = local_fs.isdir(source_path)
    if is_dir and not recursive:
        raise DataChainError("Cannot copy directory without --recursive")
    return is_dir


def build_file_paths(
    source_path: str,
    destination_path: str,
    local_fs: "AbstractFileSystem",
    is_dir: bool,
):
    """Build mapping of destination paths to source paths."""
    from datachain.client.fsspec import Client

    client = Client.get_implementation(destination_path)
    _, subpath = client.split_url(destination_path)

    if is_dir:
        folder_name = os.path.basename(source_path)
        return {
            os.path.join(subpath, folder_name, os.path.relpath(path, source_path)): path
            for path in local_fs.find(source_path)
        }

    destination_path = (
        os.path.join(subpath, os.path.basename(source_path))
        if destination_path.endswith(("/", "\\")) or not subpath
        else subpath
    )
    return {destination_path: source_path}
