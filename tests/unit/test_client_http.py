from unittest.mock import Mock

import pytest

from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient, HTTPSClient


def test_protocol_detection_https():
    client_class = Client.get_implementation("https://example.com/file.txt")
    assert client_class == HTTPSClient

    client_class = Client.get_implementation("http://example.com/file.txt")
    assert client_class == HTTPClient


def test_split_url_https():
    """Test URL splitting for HTTPS URLs"""
    domain, path = HTTPClient.split_url("https://example.com/path/to/file.txt")
    assert domain == "example.com"
    assert path == "path/to/file.txt"


def test_is_root_url():
    """Test root URL detection"""
    assert HTTPClient.is_root_url("https://example.com")
    assert HTTPClient.is_root_url("https://example.com/")
    assert not HTTPClient.is_root_url("https://example.com/path")
    assert not HTTPClient.is_root_url("https://example.com?query=1")
    assert not HTTPClient.is_root_url("https://example.com#fragment")


def test_from_name_with_https():
    """Test creating client from HTTPS URL"""
    cache = Mock(spec=Cache)
    client = HTTPSClient.from_name("https://example.com/path", cache, {})
    assert client.protocol == "https"
    assert client.name == "example.com/path"
    assert client.PREFIX == "https://"


def test_from_name_with_http():
    """Test creating client from HTTP URL"""
    cache = Mock(spec=Cache)
    client = HTTPClient.from_name("http://example.com/path", cache, {})
    assert client.protocol == "http"
    assert client.name == "example.com/path"
    assert client.PREFIX == "http://"


def test_get_full_path_http():
    """Test full path construction for HTTP"""
    cache = Mock(spec=Cache)
    client = HTTPClient("example.com:8080", {}, cache)

    assert client.get_full_path("file.txt") == "http://example.com:8080/file.txt"


def test_upload_raises_not_implemented():
    """Test that upload raises NotImplementedError"""
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError, match="HTTP/HTTPS client is read-only"):
        client.upload(b"data", "path/to/file.txt")


async def test_fetch_dir_always_raises():
    """Test that _fetch_dir always raises NotImplementedError for HTTP/HTTPS"""
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    # Should always raise NotImplementedError
    with pytest.raises(
        NotImplementedError,
        match="Cannot download file from https://example.com/any-path",
    ):
        await client._fetch_dir("any-path", None, None)

    # Test with different paths - all should raise
    with pytest.raises(
        NotImplementedError, match="Cannot download file from https://example.com"
    ):
        await client._fetch_dir("", None, None)

    with pytest.raises(
        NotImplementedError,
        match="Cannot download file from https://example.com/file.txt",
    ):
        await client._fetch_dir("file.txt", None, None)
