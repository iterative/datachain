from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast
from urllib.parse import urlparse

from fsspec.implementations.http import HTTPFileSystem

from datachain.lib.file import File

from .fsspec import Client

if TYPE_CHECKING:
    from datachain.cache import Cache


class HTTPClient(Client):
    """
    HTTP/HTTPS client for read-only access to web resources.
    Supports both HTTP and HTTPS protocols.
    """

    FS_CLASS = HTTPFileSystem
    PREFIX: ClassVar[str] = ""  # Will be set dynamically based on protocol
    protocol: ClassVar[str] = ""  # Will be set dynamically

    def __init__(
        self,
        name: str,
        fs_kwargs: dict[str, Any],
        cache: "Cache",
        protocol: str = "https",
    ) -> None:
        super().__init__(name, fs_kwargs, cache)
        self.protocol = protocol
        self.PREFIX = f"{protocol}://"

    @classmethod
    def create_fs(cls, **kwargs) -> HTTPFileSystem:
        # Configure HTTPFileSystem options
        kwargs.setdefault("simple_links", True)
        kwargs.setdefault("same_scheme", True)
        kwargs.setdefault("cache_type", "bytes")

        # HTTPFileSystem doesn't support version_aware, remove it if present
        kwargs.pop("version_aware", None)

        # Create filesystem without calling super() to avoid version_aware
        fs = cls.FS_CLASS(**kwargs)
        fs.invalidate_cache()
        return cast("HTTPFileSystem", fs)

    @classmethod
    def from_name(
        cls,
        name: str,
        cache: "Cache",
        kwargs: dict[str, Any],
    ) -> "HTTPClient":
        # Determine protocol from the name if it includes it
        parsed = urlparse(name)
        protocol = parsed.scheme if parsed.scheme in ("http", "https") else "https"

        # Extract just the host/path part without protocol
        if parsed.scheme:
            name = parsed.netloc + parsed.path

        return cls(name, kwargs, cache, protocol=protocol)

    @classmethod
    def split_url(cls, url: str) -> tuple[str, str]:
        """
        Split HTTP/HTTPS URL into domain (bucket equivalent) and path.
        Examples:
            https://example.com/path/to/file.txt -> (example.com, path/to/file.txt)
            http://example.com:8080/file.txt -> (example.com:8080, file.txt)
        """
        parsed = urlparse(url)

        # Domain includes host and port if present
        domain = parsed.netloc

        # Path without leading slash
        path = parsed.path.lstrip("/")

        # Include query and fragment if present
        if parsed.query:
            path += f"?{parsed.query}"
        if parsed.fragment:
            path += f"#{parsed.fragment}"

        return domain, path

    @classmethod
    def get_uri(cls, name: str) -> "StorageURI":
        from datachain.dataset import StorageURI

        # If name doesn't have protocol, default to https
        if not name.startswith(("http://", "https://")):
            return StorageURI(f"https://{name}")
        return StorageURI(name)

    @classmethod
    def is_root_url(cls, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.path in ("", "/") and not parsed.query and not parsed.fragment

    def get_full_path(self, rel_path: str, version_id: Optional[str] = None) -> str:
        """
        Construct full HTTP/HTTPS URL from relative path.
        Note: version_id is ignored as HTTP doesn't support versioning.
        """
        if version_id:
            # HTTP doesn't support versioning, ignore it silently
            pass

        # Check if self.name already contains the protocol (shouldn't happen but defensive)
        if self.name.startswith(("http://", "https://")):
            # Name already has protocol, use it as-is
            base_url = self.name
        else:
            # Check if rel_path already contains the full URL path including domain
            # This happens when File has source="https://" and path="domain.com/path/file"
            if rel_path and "/" in rel_path:
                # Check if the first part looks like a domain
                first_part = rel_path.split("/")[0]
                if "." in first_part and not first_part.startswith("."):
                    # This looks like domain.com/path/file format
                    # Just prepend the protocol
                    return f"{self.protocol}://{rel_path}"

            # Normal case: prepend protocol and name
            base_url = f"{self.protocol}://{self.name}"

        if rel_path:
            # Ensure single slash between base and path
            if not base_url.endswith("/") and not rel_path.startswith("/"):
                base_url += "/"
            full_url = base_url + rel_path
        else:
            full_url = base_url

        return full_url

    def url(self, path: str, expires: int = 3600, **kwargs) -> str:
        """
        Generate URL for the given path.
        Note: HTTP URLs don't support signed/expiring URLs.
        """
        return self.get_full_path(path, kwargs.pop("version_id", None))

    def info_to_file(self, v: dict[str, Any], path: str) -> File:
        """
        Convert HTTP file info to DataChain File object.
        """
        # Extract ETag if available (remove quotes if present)
        etag = v.get("ETag", "").strip('"')

        # Parse last modified time
        last_modified = v.get("last_modified")
        if last_modified:
            if isinstance(last_modified, str):
                # Try to parse string date
                try:
                    from email.utils import parsedate_to_datetime
                    last_modified = parsedate_to_datetime(last_modified)
                except (ValueError, TypeError):
                    last_modified = datetime.now(timezone.utc)
            elif isinstance(last_modified, (int, float)):
                # Timestamp
                last_modified = datetime.fromtimestamp(last_modified, timezone.utc)
        else:
            last_modified = datetime.now(timezone.utc)

        return File(
            source=self.uri,
            path=path,
            size=v.get("size", 0),
            etag=etag,
            version="",  # HTTP doesn't support versioning
            is_latest=True,  # Always latest for HTTP
            last_modified=last_modified,
        )

    def upload(self, data: bytes, path: str) -> "File":
        """
        Upload is not supported for HTTP/HTTPS protocol (read-only).
        """
        raise NotImplementedError(
            "HTTP/HTTPS client is read-only. Upload operations are not supported."
        )

    def get_file_info(self, path: str, version_id: Optional[str] = None) -> "File":
        """
        Get file info for HTTP/HTTPS file.
        Note: version_id is ignored as HTTP doesn't support versioning.
        """
        # HTTP doesn't support versioning, don't pass version_id
        info = self.fs.info(self.get_full_path(path))
        return self.info_to_file(info, path)

    def open_object(
        self, file: "File", use_cache: bool = True, cb = None
    ):
        """
        Open an HTTP/HTTPS file.
        Note: HTTP doesn't support versioning, so file.version is ignored.
        """
        from datachain.client.fileslice import FileWrapper
        
        if use_cache and (cache_path := self.cache.get_path(file)):
            return open(cache_path, mode="rb")
        
        assert not file.location
        # Don't pass version to fs.open for HTTP
        return FileWrapper(
            self.fs.open(self.get_full_path(file.get_path_normalized())),
            cb or (lambda x: None),
        )

    async def get_file(self, lpath, rpath, callback, version_id: Optional[str] = None):
        """
        Download file from HTTP/HTTPS.
        Note: version_id is ignored as HTTP doesn't support versioning.
        """
        # Don't pass version_id to HTTP filesystem
        return await self.fs._get_file(lpath, rpath, callback=callback)

    async def _fetch_dir(
        self, prefix: str, pbar, result_queue
    ) -> set[str]:
        """
        Fetch directory listing via HTTP.
        This uses the HTTPFileSystem's HTML parsing capabilities.
        """
        full_url = self.get_full_path(prefix)

        try:
            # First, check if this might be a direct file URL (not a directory)
            # Try to get file info directly
            try:
                info = await self.fs._info(full_url)
                if info.get("type") == "file":
                    # This is a direct file URL, not a directory
                    # Extract the filename from the URL
                    parsed = urlparse(full_url)
                    filename = parsed.path.split("/")[-1] if parsed.path else "file"
                    
                    # Create file info for this single file
                    file = self.info_to_file(info, prefix if prefix else filename)
                    await result_queue.put([file])
                    pbar.update(1)
                    return set()  # No subdirectories for a single file
            except (FileNotFoundError, OSError):
                # Not a direct file, try directory listing
                pass

            # Try directory listing
            infos = await self.fs._ls(full_url, detail=True)

            files = []
            subdirs = set()

            for info in infos:
                # Extract relative path from the full URL
                file_url = info["name"]
                parsed = urlparse(file_url)

                # Get path relative to our base
                if parsed.netloc == self.name:
                    rel_path = parsed.path.lstrip("/")

                    # Include query and fragment if present
                    if parsed.query:
                        rel_path += f"?{parsed.query}"
                    if parsed.fragment:
                        rel_path += f"#{parsed.fragment}"
                else:
                    # Different domain, skip
                    continue

                # Skip if it's the same as prefix (self-reference)
                if prefix.strip("/") == rel_path.strip("/"):
                    continue

                if info["type"] == "directory":
                    subdirs.add(rel_path)
                else:
                    files.append(self.info_to_file(info, rel_path))

            if files:
                await result_queue.put(files)

            found_count = len(subdirs) + len(files)
            pbar.update(found_count)

            return subdirs

        except (FileNotFoundError, OSError):
            # HTTP directory listing might not be available
            # Return empty set to indicate no subdirectories found
            return set()

