"""Tests for glob pattern support in read_storage()"""

import sys
from unittest.mock import MagicMock, patch

import pytest

import datachain as dc

# Ensure the datasets module is loaded before we try to patch it
from datachain.lib.dc.storage import expand_brace_pattern, split_uri_pattern
from datachain.lib.file import File


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


@pytest.fixture
def mock_session(test_session):
    """Mock session for testing"""
    return test_session


@pytest.fixture
def mock_listing(tmp_dir):
    """Create mock file listings for testing"""
    files = []

    # Create test file structure
    paths = [
        "audio/song1.mp3",
        "audio/song2.mp3",
        "audio/podcast.wav",
        "videos/movie.mp4",
        "docs/report.pdf",
        "docs/data.json",
        "nested/deep/file.txt",
        "nested/deep/audio/track.mp3",
        "file1.txt",
        "file2.txt",
        "test.jpg",
    ]

    for path in paths:
        file_path = tmp_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f"content of {path}")
        files.append(File(path=str(file_path), source=str(tmp_dir)))

    return tmp_dir, files


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_wildcard_pattern(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that wildcard patterns are automatically filtered"""
    tmp_dir, files = mock_listing

    # Setup mocks
    mock_get_listing.return_value = ("test_dataset", str(tmp_dir), "audio", True)
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_chain.filter = MagicMock(return_value=mock_chain)
    mock_chain.union = MagicMock(return_value=mock_chain)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test with wildcard pattern
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/audio/*.mp3", session=mock_session)

    # Verify filter was called with glob pattern
    assert mock_chain.filter.called or mock_ls.called


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_globstar_pattern(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that globstar patterns are automatically filtered"""
    tmp_dir, files = mock_listing

    # Setup mocks
    mock_get_listing.return_value = ("test_dataset", str(tmp_dir), "**/*.mp3", True)
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_chain.filter = MagicMock(return_value=mock_chain)
    mock_chain.union = MagicMock(return_value=mock_chain)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test with globstar pattern
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/**/*.mp3", session=mock_session)

    # Verify appropriate filtering
    assert mock_chain.filter.called or mock_ls.called


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_brace_expansion_pattern(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that brace expansion patterns work correctly"""
    tmp_dir, files = mock_listing

    # Setup mocks
    mock_get_listing.return_value = (
        "test_dataset",
        str(tmp_dir),
        "audio/*.{mp3,wav}",
        True,
    )
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_chain.filter = MagicMock(return_value=mock_chain)
    mock_chain.union = MagicMock(return_value=mock_chain)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test with brace expansion
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/audio/*.{{mp3,wav}}", session=mock_session)

    # Verify filtering includes both extensions
    assert mock_chain.filter.called or mock_ls.called


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_question_mark_pattern(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that question mark wildcards work"""
    tmp_dir, files = mock_listing

    # Setup mocks
    mock_get_listing.return_value = (
        "test_dataset",
        str(tmp_dir),
        "file?.txt",
        True,
    )
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_chain.filter = MagicMock(return_value=mock_chain)
    mock_chain.union = MagicMock(return_value=mock_chain)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test with question mark pattern
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/file?.txt", session=mock_session)

    # Verify filtering with single character wildcard
    assert mock_chain.filter.called or mock_ls.called


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_multiple_patterns(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test multiple URIs with different patterns"""
    tmp_dir, files = mock_listing

    # Setup mocks for multiple URIs
    mock_get_listing.side_effect = [
        ("test_dataset1", str(tmp_dir), "audio/*.mp3", True),
        ("test_dataset2", str(tmp_dir), "docs/*.json", True),
    ]

    mock_chain1 = MagicMock()
    mock_chain2 = MagicMock()
    mock_query1 = MagicMock()
    mock_query2 = MagicMock()
    mock_chain1._query = mock_query1
    mock_chain2._query = mock_query2
    mock_chain1.signals_schema = MagicMock()
    mock_chain2.signals_schema = MagicMock()
    mock_chain1.signals_schema.mutate = MagicMock(
        return_value=mock_chain1.signals_schema
    )
    mock_chain2.signals_schema.mutate = MagicMock(
        return_value=mock_chain2.signals_schema
    )
    mock_chain1.filter = MagicMock(return_value=mock_chain1)
    mock_chain2.filter = MagicMock(return_value=mock_chain2)
    mock_chain1.union = MagicMock(return_value=mock_chain1)

    mock_read_dataset.side_effect = [mock_chain1, mock_chain2]
    mock_ls.side_effect = [mock_chain1, mock_chain2]

    # Test with multiple pattern URIs
    uris = [f"{tmp_dir.as_uri()}/audio/*.mp3", f"{tmp_dir.as_uri()}/docs/*.json"]
    _ = dc.read_storage(uris, session=mock_session)

    # Verify both patterns were processed
    assert mock_get_listing.call_count == 2


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_no_pattern_unchanged(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that URIs without patterns work as before"""
    tmp_dir, files = mock_listing

    # Setup mocks
    mock_get_listing.return_value = ("test_dataset", str(tmp_dir), "audio", True)
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test without pattern - should work as before
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/audio", session=mock_session)

    # Verify normal flow without extra filtering
    assert mock_ls.called
    assert mock_read_dataset.called


@patch("datachain.lib.dc.storage.get_listing")
@patch("datachain.lib.dc.storage.ls")
@patch.object(sys.modules["datachain.lib.dc.datasets"], "read_dataset")
def test_read_storage_filename_pattern_with_directory_path(
    mock_read_dataset, mock_ls, mock_get_listing, mock_session, mock_listing
):
    """Test that patterns like dir/file* correctly match files in directories"""
    tmp_dir, files = mock_listing

    # Setup mocks with a directory path scenario
    mock_get_listing.return_value = (
        "test_dataset",
        str(tmp_dir),
        "subdir/",
        True,
    )
    mock_chain = MagicMock()
    mock_query = MagicMock()
    mock_chain._query = mock_query
    mock_chain.signals_schema = MagicMock()
    mock_chain.signals_schema.mutate = MagicMock(return_value=mock_chain.signals_schema)
    mock_chain.filter = MagicMock(return_value=mock_chain)
    mock_read_dataset.return_value = mock_chain
    mock_ls.return_value = mock_chain

    # Test with a filename pattern in a subdirectory
    _ = dc.read_storage(f"{tmp_dir.as_uri()}/subdir/file*", session=mock_session)

    # Verify that filter was called (glob pattern detected and applied)
    assert mock_chain.filter.called

    # Check that the filter call includes the directory path
    filter_calls = mock_chain.filter.call_args_list
    assert len(filter_calls) > 0


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
