import mimetypes
import os.path
from typing import TYPE_CHECKING, Optional

import requests

from datachain.cli.commands.storage.utils import build_file_paths, validate_upload_args
from datachain.client.fsspec import Client
from datachain.error import DataChainError
from datachain.lib.data_model import DataModel
from datachain.lib.dc.values import read_values
from datachain.remote.studio import StudioClient

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


class UploadFileInfo(DataModel):
    dest_path: str
    src_path: str


class DownloadFileInfo(DataModel):
    filename: str
    url: str


def upload_to_storage(
    source_path: str,
    destination_path: str,
    team: Optional[str] = None,
    recursive: bool = False,
    local_fs: "AbstractFileSystem" = None,
):
    studio_client = StudioClient(team=team)

    is_dir = validate_upload_args(source_path, recursive, local_fs)
    file_paths = build_file_paths(source_path, destination_path, local_fs, is_dir)
    response = _get_presigned_urls(studio_client, destination_path, file_paths)

    download_chain = read_values(
        info=[
            UploadFileInfo(dest_path=dest_path, src_path=src_path)
            for dest_path, src_path in file_paths.items()
        ],
    )

    download_chain = (
        download_chain.settings(cache=True, parallel=True).map(
            download_result=lambda info: _upload_single_file(
                info.dest_path, info.src_path, response, local_fs
            )
        )
    ).exec()

    print(f"Successfully uploaded {len(file_paths)} file(s)")
    return file_paths


def download_from_storage(
    source_path: str,
    destination_path: str,
    team: Optional[str] = None,
    local_fs: "AbstractFileSystem" = None,
):
    studio_client = StudioClient(team=team)
    source_client = Client.get_implementation(source_path)
    _, src_subpath = source_client.split_url(source_path)

    response = studio_client.download_url(source_path)
    if not response.ok or not response.data:
        raise DataChainError(response.message)

    urls = response.data.get("urls")
    if not urls:
        raise DataChainError("No download URL found")

    is_dest_dir = local_fs.isdir(destination_path) or destination_path.endswith(
        ("/", "\\")
    )

    def _calculate_out_path(info: DownloadFileInfo) -> str:
        filename = info.filename.removeprefix(src_subpath).removeprefix("/")
        if not filename:
            filename = os.path.basename(src_subpath)
            out_path = (
                os.path.join(destination_path, filename)
                if is_dest_dir
                else destination_path
            )
        else:
            out_path = os.path.join(destination_path, filename)
        return out_path

    download_chain = read_values(
        info=[
            DownloadFileInfo(filename=filename, url=url)
            for filename, url in urls.items()
        ],
    )
    download_chain = (
        download_chain.settings(cache=True, parallel=10)
        .map(
            out_path=_calculate_out_path,
        )
        .map(
            upload_result=lambda info, out_path: _download_single_file(
                info.url, out_path, local_fs
            )
        )
    ).exec()

    print(f"Successfully downloaded {len(urls)} file(s)")


def _download_single_file(
    url: str, out_path: str, local_fs: "AbstractFileSystem"
) -> str:
    local_fs.makedirs(os.path.dirname(out_path), exist_ok=True)
    with requests.get(url, timeout=3600, stream=True) as download_response:
        download_response.raise_for_status()
        with local_fs.open(out_path, "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

    return out_path


def copy_inside_storage(
    source_path: str,
    destination_path: str,
    team: Optional[str] = None,
    recursive: bool = False,
):
    client = StudioClient(team=team)

    response = client.copy_storage_file(
        source_path,
        destination_path,
        recursive=recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Copied {source_path} to {destination_path}")


def _get_presigned_urls(
    studio_client: "StudioClient", destination_path: str, file_paths: dict
):
    """Get presigned URLs for file uploads."""
    response = studio_client.batch_presigned_urls(
        destination_path,
        {dest: str(mimetypes.guess_type(src)[0]) for dest, src in file_paths.items()},
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
    form_data["Content-Type"] = str(content_type)

    with local_fs.open(source_path, "rb") as f:
        form_data["file"] = (
            os.path.basename(source_path),
            f,
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
        return requests.request(
            method,
            upload_url,
            data=f,
            headers={
                **headers,
                "Content-Type": str(mimetypes.guess_type(source_path)[0]),
            },
            timeout=3600,
        )


def _upload_single_file(
    dest_path: str,
    source_path: str,
    response: dict,
    local_fs: "AbstractFileSystem",
):
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
