import mimetypes
import os.path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

from datachain.error import DataChainError

if TYPE_CHECKING:
    from argparse import Namespace

    from fsspec import AbstractFileSystem


def get_studio_client(args: "Namespace"):
    from datachain.config import Config
    from datachain.remote.studio import StudioClient

    if Config().read().get("studio", {}).get("token"):
        return StudioClient(team=args.team)

    raise DataChainError("Not logged in to Studio. Log in with 'datachain auth login'.")


def upload_to_storage(args: "Namespace", local_fs: "AbstractFileSystem"):
    studio_client = get_studio_client(args)

    is_dir = _validate_upload_args(args, local_fs)
    file_paths = _build_file_paths(args, local_fs, is_dir)
    response = _get_presigned_urls(studio_client, args.destination_path, file_paths)

    for dest_path, source_path in file_paths.items():
        _upload_single_file(
            dest_path,
            source_path,
            response,
            local_fs,
        )

    _save_upload_log(studio_client, args.destination_path, file_paths, local_fs)
    print(f"Successfully uploaded {len(file_paths)} file(s)")


def download_from_storage(args: "Namespace", local_fs: "AbstractFileSystem"):
    studio_client = get_studio_client(args)
    response = studio_client.download_url(args.source_path)
    if not response.ok:
        raise DataChainError(response.message)

    url = response.data.get("url")
    if not url:
        raise DataChainError("No download URL found")

    # Extract filename from URL if destination is a directory
    if local_fs.isdir(args.destination_path) or args.destination_path.endswith(
        ("/", "\\")
    ):
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        local_fs.makedirs(args.destination_path, exist_ok=True)
        destination_path = os.path.join(args.destination_path, filename)
    else:
        destination_path = args.destination_path

    # Stream download to avoid loading entire file into memory
    with requests.get(url, timeout=3600, stream=True) as download_response:
        download_response.raise_for_status()
        print("Downloading file", end="")
        with local_fs.open(destination_path, "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                print(".", end="")
        print()

    print(f"Downloaded to {destination_path}")


def copy_inside_storage(args: "Namespace"):
    client = get_studio_client(args)

    response = client.copy_storage_file(
        args.source_path,
        args.destination_path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Copied {args.source_path} to {args.destination_path}")


def _validate_upload_args(args: "Namespace", local_fs: "AbstractFileSystem"):
    """Validate upload arguments and raise appropriate errors."""
    is_dir = local_fs.isdir(args.source_path)
    if is_dir and not args.recursive:
        raise DataChainError("Cannot copy directory without --recursive")
    return is_dir


def _build_file_paths(args: "Namespace", local_fs: "AbstractFileSystem", is_dir: bool):
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


def _get_presigned_urls(studio_client, destination_path: str, file_paths: dict):
    """Get presigned URLs for file uploads."""
    response = studio_client.batch_presigned_urls(
        destination_path,
        {dest: mimetypes.guess_type(src)[0] for dest, src in file_paths.items()},
    )
    if not response.ok:
        raise DataChainError(response.message)

    return response.data


def _upload_file_s3(
    upload_url: str, url_data: dict, source_path: str, local_fs: "AbstractFileSystem"
):
    """Upload file using S3 multipart form data."""
    form_data = dict(url_data["fields"])
    content_type = mimetypes.guess_type(source_path)[0]
    form_data["Content-Type"] = content_type

    file_content = local_fs.open(source_path, "rb").read()
    form_data["file"] = (
        os.path.basename(source_path),
        file_content,
        content_type,
    )

    return requests.post(upload_url, files=form_data, timeout=3600)


def _upload_file_direct(
    upload_url: str,
    method: str,
    headers: dict,
    source_path: str,
    local_fs: "AbstractFileSystem",
):
    """Upload file using direct HTTP request."""
    with local_fs.open(source_path, "rb") as f:
        file_content = f.read()

    return requests.request(
        method,
        upload_url,
        data=file_content,
        headers={
            **headers,
            "Content-Type": mimetypes.guess_type(source_path)[0],
        },
        timeout=3600,
    )


def _upload_single_file(
    dest_path: str,
    source_path: str,
    response: dict,
    local_fs: "AbstractFileSystem",
):
    """Upload a single file using the appropriate method."""
    urls = response.get("urls", {})
    headers = response.get("headers", {})
    method = response.get("method", "PUT")

    if dest_path not in urls:
        raise DataChainError(f"No presigned URL found for {dest_path}")

    upload_url = urls[dest_path]["url"]

    if "fields" in urls[dest_path]:
        upload_response = _upload_file_s3(
            upload_url, urls[dest_path], source_path, local_fs
        )
    else:
        upload_response = _upload_file_direct(
            upload_url, method, headers, source_path, local_fs
        )

    if upload_response.status_code >= 400:
        raise DataChainError(
            f"Failed to upload {source_path} to {dest_path}. "
            f"Status: {upload_response.status_code}, "
            f"Response: {upload_response.text}"
        )

    print(f"Uploaded {source_path} to {dest_path}")


def _save_upload_log(
    studio_client,
    destination_path: str,
    file_paths: dict,
    local_fs: "AbstractFileSystem",
):
    """Save upload log to studio."""
    uploads = [
        {
            "path": dst,
            "size": local_fs.info(src).get("size", 0),
        }
        for dst, src in file_paths.items()
    ]
    studio_client.save_upload_log(destination_path, uploads)
