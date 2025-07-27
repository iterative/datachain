import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import PurePosixPath
from urllib.parse import urlparse


class AbstractUDF(ABC):
    @abstractmethod
    def process(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass


class DataChainError(Exception):
    pass


class DataChainParamsError(DataChainError):
    pass


class DataChainColumnError(DataChainParamsError):
    def __init__(self, col_name: str, msg: str):
        super().__init__(f"Error for column {col_name}: {msg}")


def normalize_col_names(col_names: Sequence[str]) -> dict[str, str]:
    """Returns normalized_name -> original_name dict."""
    gen_col_counter = 0
    new_col_names = {}
    org_col_names = set(col_names)

    for org_column in col_names:
        new_column = org_column.lower()
        new_column = re.sub("[^0-9a-z]+", "_", new_column)
        new_column = new_column.strip("_")

        generated_column = new_column

        while (
            not generated_column.isidentifier()
            or generated_column in new_col_names
            or (generated_column != org_column and generated_column in org_col_names)
        ):
            if new_column:
                generated_column = f"c{gen_col_counter}_{new_column}"
            else:
                generated_column = f"c{gen_col_counter}"
            gen_col_counter += 1

        new_col_names[generated_column] = org_column

    return new_col_names


def rebase_path(
    src_path: str,
    old_base: str,
    new_base: str,
    suffix: str = "",
    extension: str = "",
) -> str:
    """
    Rebase a file path from one base directory to another.

    Args:
        src_path: Source file path (can include URI scheme like s3://)
        old_base: Base directory to remove from src_path
        new_base: New base directory to prepend
        suffix: Optional suffix to add before file extension
        extension: Optional new file extension (without dot)

    Returns:
        str: Rebased path with new base directory

    Raises:
        ValueError: If old_base is not found in src_path
    """
    # Parse URIs to handle schemes properly
    src_parsed = urlparse(src_path)
    old_base_parsed = urlparse(old_base)
    new_base_parsed = urlparse(new_base)

    # Get the path component (without scheme)
    if src_parsed.scheme:
        src_path_only = src_parsed.netloc + src_parsed.path
    else:
        src_path_only = src_path

    if old_base_parsed.scheme:
        old_base_only = old_base_parsed.netloc + old_base_parsed.path
    else:
        old_base_only = old_base

    # Normalize paths
    src_path_norm = PurePosixPath(src_path_only).as_posix()
    old_base_norm = PurePosixPath(old_base_only).as_posix()

    # Find where old_base appears in src_path
    if old_base_norm in src_path_norm:
        # Find the index where old_base appears
        idx = src_path_norm.find(old_base_norm)
        if idx == -1:
            raise ValueError(f"old_base '{old_base}' not found in src_path")

        # Extract the relative path after old_base
        relative_start = idx + len(old_base_norm)
        # Skip leading slash if present
        if relative_start < len(src_path_norm) and src_path_norm[relative_start] == "/":
            relative_start += 1
        relative_path = src_path_norm[relative_start:]
    else:
        raise ValueError(f"old_base '{old_base}' not found in src_path")

    # Parse the filename
    path_obj = PurePosixPath(relative_path)
    stem = path_obj.stem
    current_ext = path_obj.suffix

    # Apply suffix and extension changes
    new_stem = stem + suffix if suffix else stem
    if extension:
        new_ext = f".{extension}"
    elif current_ext:
        new_ext = current_ext
    else:
        new_ext = ""

    # Build new filename
    new_name = new_stem + new_ext

    # Reconstruct path with new base
    parent = str(path_obj.parent)
    if parent == ".":
        new_relative_path = new_name
    else:
        new_relative_path = str(PurePosixPath(parent) / new_name)

    # Handle new_base URI scheme
    if new_base_parsed.scheme:
        # Has schema like s3://
        base_path = new_base_parsed.netloc + new_base_parsed.path
        base_path = PurePosixPath(base_path).as_posix()
        full_path = str(PurePosixPath(base_path) / new_relative_path)
        return f"{new_base_parsed.scheme}://{full_path}"
    # Regular path
    return str(PurePosixPath(new_base) / new_relative_path)
