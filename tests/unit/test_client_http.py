from datetime import datetime
from unittest.mock import Mock

import pytest

from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient, HTTPSClient
from datachain.lib.file import File


def test_protocol_detection_https():
    client_class = Client.get_implementation("https://example.com/file.txt")
    assert client_class == HTTPSClient

    client_class = Client.get_implementation("http://example.com/file.txt")
    assert client_class == HTTPClient


def test_split_url_https():
    domain, path = HTTPClient.split_url("https://example.com/path/to/file.txt")
    assert domain == "example.com"
    assert path == "path/to/file.txt"


def test_is_root_url():
    assert HTTPClient.is_root_url("https://example.com")
    assert HTTPClient.is_root_url("https://example.com/")
    assert not HTTPClient.is_root_url("https://example.com/path")
    assert not HTTPClient.is_root_url("https://example.com?query=1")
    assert not HTTPClient.is_root_url("https://example.com#fragment")


def test_from_name_with_https():
    cache = Mock(spec=Cache)
    client = HTTPSClient.from_name("https://example.com/path", cache, {})
    assert client.protocol == "https"
    assert client.name == "example.com/path"
    assert client.PREFIX == "https://"


def test_from_name_with_http():
    cache = Mock(spec=Cache)
    client = HTTPClient.from_name("http://example.com/path", cache, {})
    assert client.protocol == "http"
    assert client.name == "example.com/path"
    assert client.PREFIX == "http://"


def test_get_full_path_http():
    cache = Mock(spec=Cache)
    client = HTTPClient("example.com:8080", {}, cache)

    assert client.get_full_path("file.txt") == "http://example.com:8080/file.txt"


def test_upload_raises_not_implemented():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    with pytest.raises(NotImplementedError, match="HTTP/HTTPS client is read-only"):
        client.upload(b"data", "path/to/file.txt")


def test_create_fs():
    from fsspec.implementations.http import HTTPFileSystem

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs(version_aware=True)
    assert isinstance(fs, HTTPFileSystem)
    assert not hasattr(fs, "version_aware")

    fs = HTTPClient.create_fs(timeout=30, headers={"User-Agent": "test"})
    assert isinstance(fs, HTTPFileSystem)

    fs = HTTPClient.create_fs()
    assert isinstance(fs, HTTPFileSystem)


def test_info_to_file():
    cache = Mock(spec=Cache)
    client = HTTPSClient("example.com", {}, cache)

    info = {
        "size": 1024,
        "ETag": '"abc123"',
        "last_modified": "Wed, 12 Oct 2024 07:28:00 GMT",
    }
    file = client.info_to_file(info, "path/to/file.txt")

    assert isinstance(file, File)
    assert file.path == "path/to/file.txt"
    assert file.size == 1024
    assert file.etag == "abc123"
    assert file.version == ""
    assert file.is_latest is True
    assert isinstance(file.last_modified, datetime)
