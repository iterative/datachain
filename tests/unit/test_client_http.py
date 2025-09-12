from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from datachain.cache import Cache
from datachain.client.fsspec import Client
from datachain.client.http import HTTPClient
from datachain.lib.file import File


class TestHTTPClient:
    def test_protocol_detection_https(self):
        """Test that HTTPS URLs are correctly detected"""
        client_class = Client.get_implementation("https://example.com/file.txt")
        assert client_class == HTTPClient

    def test_protocol_detection_http(self):
        """Test that HTTP URLs are correctly detected"""
        client_class = Client.get_implementation("http://example.com/file.txt")
        assert client_class == HTTPClient

    def test_split_url_https(self):
        """Test URL splitting for HTTPS URLs"""
        domain, path = HTTPClient.split_url("https://example.com/path/to/file.txt")
        assert domain == "example.com"
        assert path == "path/to/file.txt"

    def test_split_url_http_with_port(self):
        """Test URL splitting for HTTP URLs with port"""
        domain, path = HTTPClient.split_url("http://example.com:8080/file.txt")
        assert domain == "example.com:8080"
        assert path == "file.txt"

    def test_split_url_with_query(self):
        """Test URL splitting with query parameters"""
        domain, path = HTTPClient.split_url("https://example.com/file.txt?param=value")
        assert domain == "example.com"
        assert path == "file.txt?param=value"

    def test_split_url_with_fragment(self):
        """Test URL splitting with fragment"""
        domain, path = HTTPClient.split_url("https://example.com/file.txt#section")
        assert domain == "example.com"
        assert path == "file.txt#section"

    def test_split_url_root(self):
        """Test URL splitting for root URL"""
        domain, path = HTTPClient.split_url("https://example.com/")
        assert domain == "example.com"
        assert path == ""

    def test_get_uri_with_protocol(self):
        """Test get_uri with protocol already included"""
        uri = HTTPClient.get_uri("https://example.com")
        assert str(uri) == "https://example.com"

    def test_get_uri_without_protocol(self):
        """Test get_uri defaults to HTTPS when protocol is missing"""
        uri = HTTPClient.get_uri("example.com")
        assert str(uri) == "https://example.com"

    def test_is_root_url(self):
        """Test root URL detection"""
        assert HTTPClient.is_root_url("https://example.com")
        assert HTTPClient.is_root_url("https://example.com/")
        assert not HTTPClient.is_root_url("https://example.com/path")
        assert not HTTPClient.is_root_url("https://example.com?query=1")
        assert not HTTPClient.is_root_url("https://example.com#fragment")

    def test_from_name_with_https(self):
        """Test creating client from HTTPS URL"""
        cache = Mock(spec=Cache)
        client = HTTPClient.from_name("https://example.com/path", cache, {})
        assert client.protocol == "https"
        assert client.name == "example.com/path"
        assert client.PREFIX == "https://"

    def test_from_name_with_http(self):
        """Test creating client from HTTP URL"""
        cache = Mock(spec=Cache)
        client = HTTPClient.from_name("http://example.com/path", cache, {})
        assert client.protocol == "http"
        assert client.name == "example.com/path"
        assert client.PREFIX == "http://"

    def test_from_name_without_protocol(self):
        """Test creating client defaults to HTTPS"""
        cache = Mock(spec=Cache)
        client = HTTPClient.from_name("example.com", cache, {})
        assert client.protocol == "https"
        assert client.name == "example.com"

    def test_get_full_path(self):
        """Test full path construction"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        assert client.get_full_path("file.txt") == "https://example.com/file.txt"
        assert (
            client.get_full_path("path/to/file.txt")
            == "https://example.com/path/to/file.txt"
        )
        assert client.get_full_path("") == "https://example.com"

        # Test case when path contains full domain (e.g., File with source="https://")
        client_empty = HTTPClient("", {}, cache, protocol="https")
        full_domain_path = "d37ci6vzurychx.cloudfront.net/trip-data/file.parquet"
        assert (
            client_empty.get_full_path(full_domain_path)
            == f"https://{full_domain_path}"
        )

        # Test case when client name already has protocol (defensive check)
        client_with_protocol = HTTPClient(
            "https://example.com", {}, cache, protocol="https"
        )
        assert (
            client_with_protocol.get_full_path("file.txt")
            == "https://example.com/file.txt"
        )
        # Should not create double https://
        assert "https://https://" not in client_with_protocol.get_full_path("file.txt")

    def test_get_full_path_http(self):
        """Test full path construction for HTTP"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com:8080", {}, cache, protocol="http")

        assert client.get_full_path("file.txt") == "http://example.com:8080/file.txt"

    def test_get_full_path_ignores_version(self):
        """Test that version_id is ignored in get_full_path"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        # Version should be ignored for HTTP
        assert (
            client.get_full_path("file.txt", version_id="v123")
            == "https://example.com/file.txt"
        )

    def test_url_method(self):
        """Test URL generation (should be same as get_full_path for HTTP)"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        url = client.url("path/to/file.txt")
        assert url == "https://example.com/path/to/file.txt"

        # Expires parameter should be ignored
        url = client.url("file.txt", expires=7200)
        assert url == "https://example.com/file.txt"

    def test_info_to_file(self):
        """Test converting HTTP file info to File object"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        info = {
            "size": 1024,
            "ETag": '"abc123"',
            "last_modified": 1609459200,  # 2021-01-01 00:00:00 UTC
            "type": "file",
        }

        file = client.info_to_file(info, "path/to/file.txt")

        assert isinstance(file, File)
        assert file.path == "path/to/file.txt"
        assert file.size == 1024
        assert file.etag == "abc123"  # Quotes should be stripped
        assert file.version == ""  # No versioning in HTTP
        assert file.is_latest == True
        assert isinstance(file.last_modified, datetime)

    def test_info_to_file_with_string_date(self):
        """Test converting HTTP file info with string date"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        info = {
            "size": 2048,
            "ETag": 'W/"xyz789"',
            "last_modified": "Fri, 01 Jan 2021 00:00:00 GMT",
            "type": "file",
        }

        file = client.info_to_file(info, "file.txt")

        assert file.size == 2048
        assert (
            file.etag == 'W/"xyz789'
        )  # Complex ETags preserved after stripping outer quotes
        assert isinstance(file.last_modified, datetime)

    def test_info_to_file_no_etag(self):
        """Test converting HTTP file info without ETag"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        info = {"size": 512, "type": "file"}

        file = client.info_to_file(info, "file.txt")

        assert file.etag == ""
        assert file.size == 512

    def test_upload_raises_not_implemented(self):
        """Test that upload raises NotImplementedError"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        with pytest.raises(NotImplementedError, match="HTTP/HTTPS client is read-only"):
            client.upload(b"data", "path/to/file.txt")

    @pytest.mark.asyncio
    async def test_fetch_dir_basic(self):
        """Test directory fetching via HTTP"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        # Mock the filesystem
        mock_fs = AsyncMock()
        mock_fs._ls = AsyncMock(
            return_value=[
                {
                    "name": "https://example.com/file1.txt",
                    "size": 100,
                    "type": "file",
                    "ETag": '"etag1"',
                },
                {
                    "name": "https://example.com/subdir/",
                    "type": "directory",
                },
                {
                    "name": "https://example.com/file2.html",
                    "size": 200,
                    "type": "file",
                    "ETag": '"etag2"',
                },
            ]
        )
        client._fs = mock_fs

        # Mock progress bar and result queue
        pbar = Mock()
        result_queue = AsyncMock()

        subdirs = await client._fetch_dir("", pbar, result_queue)

        # Check that files were added to queue
        result_queue.put.assert_called_once()
        files = result_queue.put.call_args[0][0]
        assert len(files) == 2
        assert files[0].path == "file1.txt"
        assert files[1].path == "file2.html"

        # Check subdirectories
        assert subdirs == {"subdir/"}

        # Check progress bar was updated
        pbar.update.assert_called_with(3)  # 2 files + 1 directory

    @pytest.mark.asyncio
    async def test_fetch_dir_with_prefix(self):
        """Test directory fetching with prefix"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        mock_fs = AsyncMock()
        mock_fs._ls = AsyncMock(
            return_value=[
                {
                    "name": "https://example.com/data/file.json",
                    "size": 300,
                    "type": "file",
                }
            ]
        )
        client._fs = mock_fs

        pbar = Mock()
        result_queue = AsyncMock()

        subdirs = await client._fetch_dir("data", pbar, result_queue)

        # Check the correct URL was requested
        mock_fs._ls.assert_called_with("https://example.com/data", detail=True)

        # Check files were processed correctly
        result_queue.put.assert_called_once()
        files = result_queue.put.call_args[0][0]
        assert len(files) == 1
        assert files[0].path == "data/file.json"

    @pytest.mark.asyncio
    async def test_fetch_dir_error_handling(self):
        """Test directory fetching error handling"""
        cache = Mock(spec=Cache)
        client = HTTPClient("example.com", {}, cache, protocol="https")

        # Mock filesystem to raise an error (e.g., 404)
        mock_fs = AsyncMock()
        mock_fs._ls = AsyncMock(side_effect=FileNotFoundError("404 Not Found"))
        client._fs = mock_fs

        pbar = Mock()
        result_queue = AsyncMock()

        # Should return empty set on error
        subdirs = await client._fetch_dir("nonexistent", pbar, result_queue)
        assert subdirs == set()

    def test_create_fs_defaults(self):
        """Test that create_fs sets proper defaults for HTTPFileSystem"""
        # Test that version_aware is not passed to HTTPFileSystem
        result = HTTPClient.create_fs()

        # Just verify it returns an HTTPFileSystem instance
        from fsspec.implementations.http import HTTPFileSystem

        assert isinstance(result, HTTPFileSystem)

    def test_client_initialization(self):
        """Test HTTPClient initialization"""
        cache = Mock(spec=Cache)
        client = HTTPClient(
            "example.com", {"custom_option": "value"}, cache, protocol="http"
        )

        assert client.name == "example.com"
        assert client.protocol == "http"
        assert client.PREFIX == "http://"
        assert client.fs_kwargs == {"custom_option": "value"}
        assert client.cache == cache
