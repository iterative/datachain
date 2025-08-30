"""Pattern matching utilities for storage operations.

This module contains functions for handling glob patterns, brace expansion,
and converting patterns to GLOB-compatible formats.
"""

import glob
from typing import TYPE_CHECKING, Union

from datachain.lib.listing import ls

if TYPE_CHECKING:
    from .datachain import DataChain


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
    # Check if URI contains any glob patterns
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
            if glob.has_magic(segment):
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
        if glob.has_magic(segment):
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
    Recursively expand brace patterns like *.{mp3,wav} into multiple glob patterns.
    Handles nested and multiple brace patterns.

    Args:
        pattern: Pattern that may contain brace expansion

    Returns:
        List of expanded patterns

    Examples:
        "*.{mp3,wav}" -> ["*.mp3", "*.wav"]
        "{a,b}/{c,d}" -> ["a/c", "a/d", "b/c", "b/d"]
        "*.txt" -> ["*.txt"]
        "{{a,b}}" -> ["{a}", "{b}"]  # Handle double braces
    """
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    # Handle double braces {{...}} by treating them as single braces
    # Replace {{ with a placeholder, expand, then restore
    if "{{" in pattern:
        # Replace {{ and }} with placeholders temporarily
        pattern = pattern.replace("{{", "\x00OPENBRACE\x00")
        pattern = pattern.replace("}}", "\x00CLOSEBRACE\x00")

        # Find and expand single braces
        expanded = _expand_single_braces(pattern)

        # Restore the double braces as single braces
        result = []
        for p in expanded:
            p = p.replace("\x00OPENBRACE\x00", "{")
            p = p.replace("\x00CLOSEBRACE\x00", "}")
            result.append(p)

        # Now recursively expand any remaining braces
        final_result = []
        for p in result:
            final_result.extend(expand_brace_pattern(p))

        return final_result

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
    options = pattern[start + 1 : end].split(",")

    # Generate all combinations and recursively expand
    expanded = []
    for option in options:
        combined = prefix + option.strip() + suffix
        # Recursively expand any remaining braces
        expanded.extend(_expand_single_braces(combined))

    return expanded


def expand_uri_braces(uri: str) -> list[str]:
    """
    Expand a URI that may contain brace patterns into multiple URIs.

    Args:
        uri: URI that may contain brace patterns

    Returns:
        List of URIs with all brace patterns expanded

    Examples:
        "s3://bucket/{a,b}/*.txt" -> ["s3://bucket/a/*.txt", "s3://bucket/b/*.txt"]
        "file:///root/{dir1,dir2}/**/*.{mp3,wav}" ->
            ["file:///root/dir1/**/*.mp3", "file:///root/dir1/**/*.wav",
             "file:///root/dir2/**/*.mp3", "file:///root/dir2/**/*.wav"]
    """
    return expand_brace_pattern(uri)


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
    patterns: list[str],
    list_path: str,
    use_recursive: bool,
    column: str,
) -> "DataChain":
    """Apply glob filter to a DataChain based on a single pattern.

    Since brace expansion now happens at URI level, this function
    only needs to handle single patterns.
    """
    from datachain.query.schema import Column

    chain = ls(dc, list_path, recursive=use_recursive, column=column)

    # Should only receive single patterns now (brace expansion happens earlier)
    if len(patterns) != 1:
        raise ValueError(f"Expected single pattern, got {len(patterns)}")

    pattern = patterns[0]

    # If pattern doesn't contain path separator and list_path is not empty,
    # prepend the list_path to make the pattern match correctly
    if list_path and "/" not in pattern:
        filter_pattern = f"{list_path.rstrip('/')}/{pattern}"
    else:
        filter_pattern = pattern

    # Convert globstar patterns to GLOB-compatible patterns
    glob_pattern = convert_globstar_to_glob(filter_pattern)

    return chain.filter(Column(f"{column}.path").glob(glob_pattern))
