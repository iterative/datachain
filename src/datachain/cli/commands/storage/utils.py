import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from fsspec.spec import AbstractFileSystem

from datachain.error import DataChainError


def validate_upload_args(args: "Namespace", local_fs: "AbstractFileSystem"):
    """Validate upload arguments and raise appropriate errors."""
    is_dir = local_fs.isdir(args.source_path)
    if is_dir and not args.recursive:
        raise DataChainError("Cannot copy directory without --recursive")
    return is_dir


def build_file_paths(args: "Namespace", local_fs: "AbstractFileSystem", is_dir: bool):
    """Build mapping of destination paths to source paths."""
    from datachain.client.fsspec import Client

    client = Client.get_implementation(args.destination_path)
    _, subpath = client.split_url(args.destination_path)

    if is_dir:
        return {
            os.path.join(subpath, os.path.relpath(path, args.source_path)): path
            for path in local_fs.find(args.source_path)
        }

    destination_path = (
        os.path.join(subpath, os.path.basename(args.source_path))
        if args.destination_path.endswith(("/", "\\")) or not subpath
        else subpath
    )
    return {destination_path: args.source_path}
