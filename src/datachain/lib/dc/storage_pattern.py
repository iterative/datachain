import glob
from typing import TYPE_CHECKING, Union

from datachain.client.fsspec import is_cloud_uri
from datachain.lib.listing import ls

if TYPE_CHECKING:
    from .datachain import DataChain


def validate_cloud_bucket_name(uri: str) -> None:
    """
    Validate that cloud storage bucket names don't contain glob patterns.

    Args:
        uri: URI to validate

    Raises:
        ValueError: If a cloud storage bucket name contains glob patterns
    """
    if not is_cloud_uri(uri):
        return

    # Extract bucket name (everything between :// and first /)
    if "://" in uri:
        scheme_end = uri.index("://") + 3
        path_part = uri[scheme_end:]

        # Get the bucket name (first segment)
        if "/" in path_part:
            bucket_name = path_part.split("/")[0]
        else:
            bucket_name = path_part

        # Check if bucket name contains glob patterns
        glob_chars = ["*", "?", "[", "]", "{", "}"]
        if any(char in bucket_name for char in glob_chars):
            raise ValueError(f"Glob patterns in bucket names are not supported: {uri}")


def split_uri_pattern(uri: str) -> tuple[str, Union[str, None]]:
    """
    Split a URI into base path and glob pattern.

    Args:
        uri: URI that may contain glob patterns (*, **, ?, {})

    Returns:
        Tuple of (base_uri, pattern) where pattern is None if no glob pattern found

    Examples:
        "s3://bucket/dir/*.mp3" -> ("s3://bucket/dir", "*.mp3")
        "s3://bucket/**/*.mp3" -> ("s3://bucket", "**/*.mp3")
        "s3://bucket/dir" -> ("s3://bucket/dir", None)
    """
    if not any(char in uri for char in ["*", "?", "[", "{", "}"]):
        return uri, None

    # Handle different URI schemes
    if "://" in uri:
        # Split into scheme and path
        scheme_end = uri.index("://") + 3
        scheme_part = uri[:scheme_end]
        path_part = uri[scheme_end:]

        # Find where the glob pattern starts
        path_segments = path_part.split("/")

        # Find first segment with glob pattern
        pattern_start_idx = None
        for i, segment in enumerate(path_segments):
            # Check for glob patterns including brace expansion
            if glob.has_magic(segment) or "{" in segment:
                pattern_start_idx = i
                break

        if pattern_start_idx is None:
            return uri, None

        # Split into base and pattern
        if pattern_start_idx == 0:
            # Pattern at root of bucket
            base = scheme_part + path_segments[0]
            pattern = "/".join(path_segments[1:]) if len(path_segments) > 1 else "*"
        else:
            base = scheme_part + "/".join(path_segments[:pattern_start_idx])
            pattern = "/".join(path_segments[pattern_start_idx:])

        return base, pattern
    # Local path
    path_segments = uri.split("/")

    # Find first segment with glob pattern
    pattern_start_idx = None
    for i, segment in enumerate(path_segments):
        # Check for glob patterns including brace expansion
        if glob.has_magic(segment) or "{" in segment:
            pattern_start_idx = i
            break

    if pattern_start_idx is None:
        return uri, None

    # Split into base and pattern
    base = "/".join(path_segments[:pattern_start_idx]) if pattern_start_idx > 0 else "/"
    pattern = "/".join(path_segments[pattern_start_idx:])

    return base, pattern


def should_use_recursion(pattern: str, user_recursive: bool) -> bool:
    """
    Determine if we should use recursive listing based on the pattern.

    Args:
        pattern: The glob pattern extracted from URI
        user_recursive: User's recursive preference

    Returns:
        True if recursive listing should be used

    Examples:
        "*" -> False (single level only)
        "*.mp3" -> False (single level only)
        "**/*.mp3" -> True (globstar requires recursion)
        "dir/*/file.txt" -> True (multi-level pattern)
    """
    if not user_recursive:
        # If user explicitly wants non-recursive, respect that
        return False

    # If pattern contains globstar, definitely need recursion
    if "**" in pattern:
        return True

    # If pattern contains path separators, it needs recursion
    # Single-level patterns like "*", "*.txt", "file?" should not be recursive
    return "/" in pattern


def expand_brace_pattern(pattern: str) -> list[str]:
    """
    Recursively expand brace patterns into multiple glob patterns.
    Supports:
    - Comma-separated lists: *.{mp3,wav}
    - Numeric ranges: file{1..10}
    - Zero-padded numeric ranges: file{01..10}
    - Character ranges: file{a..z}

    Args:
        pattern: Pattern that may contain brace expansion

    Returns:
        List of expanded patterns

    Examples:
        "*.{mp3,wav}" -> ["*.mp3", "*.wav"]
        "file{1..3}" -> ["file1", "file2", "file3"]
        "file{01..03}" -> ["file01", "file02", "file03"]
        "file{a..c}" -> ["filea", "fileb", "filec"]
        "{a,b}/{c,d}" -> ["a/c", "a/d", "b/c", "b/d"]
    """
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    return _expand_single_braces(pattern)


def _expand_single_braces(pattern: str) -> list[str]:
    """Helper to expand single-level braces."""
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    # Find the first complete brace pattern
    start = pattern.index("{")
    end = start
    depth = 0
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if start >= end:
        return [pattern]

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    brace_content = pattern[start + 1 : end]

    # Check if it's a range pattern (contains ..)
    if ".." in brace_content:
        options = _expand_range(brace_content)
    else:
        # Regular comma-separated list
        options = [opt.strip() for opt in brace_content.split(",")]

    # Generate all combinations and recursively expand
    expanded = []
    for option in options:
        combined = prefix + option + suffix
        # Recursively expand any remaining braces
        expanded.extend(_expand_single_braces(combined))

    return expanded


def _expand_range(range_spec: str) -> list[str]:
    """Expand range patterns like 1..10, 01..10, or a..z."""
    if ".." not in range_spec:
        return [range_spec]

    parts = range_spec.split("..")
    if len(parts) != 2:
        # Invalid range format, return as-is
        return [range_spec]

    start, end = parts[0], parts[1]

    # Check if it's a numeric range
    if start.isdigit() and end.isdigit():
        # Determine if we need zero-padding
        pad_width = max(len(start), len(end)) if start[0] == "0" or end[0] == "0" else 0
        start_num = int(start)
        end_num = int(end)

        if start_num <= end_num:
            if pad_width > 0:
                return [str(i).zfill(pad_width) for i in range(start_num, end_num + 1)]
            return [str(i) for i in range(start_num, end_num + 1)]
        # Reverse range
        if pad_width > 0:
            return [str(i).zfill(pad_width) for i in range(start_num, end_num - 1, -1)]
        return [str(i) for i in range(start_num, end_num - 1, -1)]

    # Check if it's a single character range
    if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
        start_ord = ord(start)
        end_ord = ord(end)

        if start_ord <= end_ord:
            return [chr(i) for i in range(start_ord, end_ord + 1)]
        # Reverse range
        return [chr(i) for i in range(start_ord, end_ord - 1, -1)]

    # Unknown range format, return as-is
    return [range_spec]


def convert_globstar_to_glob(filter_pattern: str) -> str:
    """Convert globstar patterns to GLOB patterns.

    Standard GLOB doesn't understand ** as recursive wildcard,
    so we need to convert patterns appropriately.

    Args:
        filter_pattern: Pattern that may contain globstars (**)

    Returns:
        GLOB-compatible pattern
    """
    if "**" not in filter_pattern:
        return filter_pattern

    parts = filter_pattern.split("/")
    globstar_positions = [i for i, p in enumerate(parts) if p == "**"]

    # Handle different cases based on number of globstars
    num_globstars = len(globstar_positions)

    if num_globstars <= 1:
        # Special case: pattern like **/* means zero or more directories
        # This is tricky because GLOB can't express "zero or more"
        # We need different handling based on the pattern structure

        if filter_pattern == "**/*":
            # Match everything
            return "*"
        if filter_pattern.startswith("**/"):
            remaining = filter_pattern[3:]
            if "/" not in remaining:
                # Pattern like **/*.ext or **/temp?.*
                # The ** means zero or more directories
                # For zero directories: pattern should be just the filename pattern
                # For one or more: pattern should be */filename
                # Since we can't OR in GLOB, we choose the more permissive option
                # that works with recursive listing
                # Special handling: if it's a simple extension pattern, match broadly
                if remaining.startswith("*."):
                    # Pattern like **/*.ext - match any file with this extension
                    # This matches *.ext at current level and deeper with recursion:
                    return remaining
                # Pattern like **/temp?.* - match as filename in subdirs
                return f"*/{remaining}"

        # Default: Zero or one globstar - simple replacement
        return filter_pattern.replace("**", "*")

    # Multiple globstars - need more careful handling
    # For patterns like **/level?/backup/**/*.ext
    # We want to match any path containing /level?/backup/ and ending with .ext

    # Find middle directories (between first and last **)
    middle_parts = []
    start_idx = globstar_positions[0] + 1
    end_idx = globstar_positions[-1]
    for i in range(start_idx, end_idx):
        if parts[i] != "**":
            middle_parts.append(parts[i])

    if not middle_parts:
        # No fixed middle parts, just use wildcards
        result = filter_pattern.replace("**", "*")
    else:
        # Create pattern that matches the middle parts
        middle_pattern = "/".join(middle_parts)
        # Get the file pattern at the end if any
        last_part = parts[-1] if parts[-1] != "**" else "*"

        # Match any path containing this pattern
        if last_part != "*":
            # Has specific file pattern
            result = f"*{middle_pattern}*{last_part}"
        else:
            result = f"*{middle_pattern}*"

    return result


def apply_glob_filter(
    dc: "DataChain",
    pattern: str,
    list_path: str,
    use_recursive: bool,
    column: str,
) -> "DataChain":
    from datachain.query.schema import Column

    chain = ls(dc, list_path, recursive=use_recursive, column=column)

    # If pattern doesn't contain path separator and list_path is not empty,
    # prepend the list_path to make the pattern match correctly
    if list_path and "/" not in pattern:
        filter_pattern = f"{list_path.rstrip('/')}/{pattern}"
    else:
        filter_pattern = pattern

    # Convert globstar patterns to GLOB-compatible patterns
    glob_pattern = convert_globstar_to_glob(filter_pattern)

    return chain.filter(Column(f"{column}.path").glob(glob_pattern))
