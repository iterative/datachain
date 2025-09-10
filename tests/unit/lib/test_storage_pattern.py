"""Tests for glob pattern support in read_storage()"""

# Ensure the datasets module is loaded before we try to patch it
from datachain.lib.dc.storage_pattern import expand_brace_pattern, split_uri_pattern


def test_split_uri_pattern_no_pattern():
    """URIs without patterns should return None for pattern"""
    assert split_uri_pattern("s3://bucket/dir") == ("s3://bucket/dir", None)
    assert split_uri_pattern("s3://bucket/dir/") == ("s3://bucket/dir/", None)
    assert split_uri_pattern("file:///home/user/data") == (
        "file:///home/user/data",
        None,
    )
    assert split_uri_pattern("/local/path") == ("/local/path", None)


def test_split_uri_pattern_wildcard():
    """Test basic wildcard patterns like *.mp3"""
    assert split_uri_pattern("s3://bucket/dir/*.mp3") == (
        "s3://bucket/dir",
        "*.mp3",
    )
    assert split_uri_pattern("s3://bucket/*.txt") == ("s3://bucket", "*.txt")
    assert split_uri_pattern("file:///data/*.json") == ("file:///data", "*.json")
    assert split_uri_pattern("/local/path/*.csv") == ("/local/path", "*.csv")


def test_split_uri_pattern_globstar():
    """Test recursive globstar patterns like **/*.mp3"""
    assert split_uri_pattern("s3://bucket/**/*.mp3") == ("s3://bucket", "**/*.mp3")
    assert split_uri_pattern("s3://bucket/dir/**/*.txt") == (
        "s3://bucket/dir",
        "**/*.txt",
    )
    assert split_uri_pattern("file:///data/**/test/*.json") == (
        "file:///data",
        "**/test/*.json",
    )


def test_split_uri_pattern_brace_expansion():
    """Test brace expansion patterns like *.{mp3,wav}"""
    assert split_uri_pattern("s3://bucket/*.{mp3,wav}") == (
        "s3://bucket",
        "*.{mp3,wav}",
    )
    assert split_uri_pattern("s3://bucket/dir/*.{jpg,png,gif}") == (
        "s3://bucket/dir",
        "*.{jpg,png,gif}",
    )
    assert split_uri_pattern("file:///data/**/*.{json,jsonl}") == (
        "file:///data",
        "**/*.{json,jsonl}",
    )


def test_split_uri_pattern_question_mark():
    """Test question mark wildcards like file?.txt"""
    assert split_uri_pattern("s3://bucket/file?.txt") == (
        "s3://bucket",
        "file?.txt",
    )
    assert split_uri_pattern("s3://bucket/dir/doc??.pdf") == (
        "s3://bucket/dir",
        "doc??.pdf",
    )
    assert split_uri_pattern("file:///data/test?/file.txt") == (
        "file:///data",
        "test?/file.txt",
    )


def test_split_uri_pattern_combined():
    """Test combinations of different glob patterns"""
    assert split_uri_pattern("s3://bucket/**/test?.{mp3,wav}") == (
        "s3://bucket",
        "**/test?.{mp3,wav}",
    )
    assert split_uri_pattern("s3://bucket/*/dir/**/*.txt") == (
        "s3://bucket",
        "*/dir/**/*.txt",
    )


def test_split_uri_pattern_edge_cases():
    """Test edge cases and special scenarios"""
    # Root level patterns
    assert split_uri_pattern("s3://bucket/*") == ("s3://bucket", "*")
    assert split_uri_pattern("s3://bucket/**") == ("s3://bucket", "**")

    # Pattern only in filename part
    assert split_uri_pattern("s3://bucket/dir/subdir/*.mp3") == (
        "s3://bucket/dir/subdir",
        "*.mp3",
    )

    # Multiple wildcards
    assert split_uri_pattern("s3://bucket/*/*.mp3") == ("s3://bucket", "*/*.mp3")


def test_expand_brace_pattern():
    """Test conversion of brace patterns to glob patterns"""
    # Single brace expansion
    assert expand_brace_pattern("*.{mp3,wav}") == ["*.mp3", "*.wav"]

    # Multiple extensions
    assert expand_brace_pattern("*.{jpg,png,gif}") == ["*.jpg", "*.png", "*.gif"]

    # Nested patterns
    assert expand_brace_pattern("**/*.{json,jsonl}") == ["**/*.json", "**/*.jsonl"]

    # No braces - return as is
    assert expand_brace_pattern("*.txt") == ["*.txt"]

    # Complex pattern
    assert expand_brace_pattern("dir/*.{mp3,wav,flac}") == [
        "dir/*.mp3",
        "dir/*.wav",
        "dir/*.flac",
    ]
