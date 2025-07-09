import mimetypes
import os.path
import sys
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

from datachain.config import Config
from datachain.error import DataChainError
from datachain.remote.studio import StudioClient

if TYPE_CHECKING:
    from argparse import Namespace

    from fsspec import AbstractFileSystem


def process_storage_command(args: "Namespace"):
    if args.cmd is None:
        print(
            f"Use 'datachain {args.command} --help' to see available options",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "rm":
        return rm_storage(args)
    if args.cmd == "mv":
        return mv_storage(args)
    if args.cmd == "cp":
        return cp_storage(args)
    raise DataChainError(f"Unknown command '{args.cmd}'.")


def _get_studio_client(args: "Namespace"):
    token = Config().read().get("studio", {}).get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain auth login'."
        )
    return StudioClient(team=args.team)


def rm_storage(args: "Namespace"):
    client = _get_studio_client(args)

    response = client.delete_storage_file(
        args.path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Deleted {args.path}")


def mv_storage(args: "Namespace"):
    client = _get_studio_client(args)

    response = client.move_storage_file(
        args.path,
        args.new_path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Moved {args.path} to {args.new_path}")


def cp_storage(args: "Namespace"):
    from datachain.client.fsspec import Client

    source_cls = Client.get_implementation(args.source_path)
    destination_cls = Client.get_implementation(args.destination_path)

    # Determine operation based on source and destination protocols
    if source_cls.protocol == "file" and destination_cls.protocol == "file":
        source_fs = source_cls.create_fs()
        source_fs.cp_file(
            args.source_path,
            args.destination_path,
        )
    elif source_cls.protocol == "file":
        source_fs = source_cls.create_fs()
        _upload_to_storage(args, source_fs)
    elif destination_cls.protocol == "file":
        destination_fs = destination_cls.create_fs()
        _download_from_storage(args, destination_fs)
    else:
        _copy_inside_storage(args)


def _upload_to_storage(args: "Namespace", local_fs: "AbstractFileSystem"):
    from datachain.client.fsspec import Client

    studio_client = _get_studio_client(args)

    is_dir = local_fs.isdir(args.source_path)
    if is_dir and not args.recursive:
        raise DataChainError("Cannot copy directory without --recursive")

    client = Client.get_implementation(args.destination_path)
    _, subpath = client.split_url(args.destination_path)

    if is_dir:
        file_paths = {
            os.path.join(subpath, os.path.relpath(path, args.source_path)): path
            for path in local_fs.find(args.source_path)
        }
    else:
        destination_path = (
            os.path.join(subpath, os.path.basename(args.source_path))
            if args.destination_path.endswith(("/", "\\")) or not subpath
            else subpath
        )
        file_paths = {destination_path: args.source_path}

    response = studio_client.batch_presigned_urls(
        args.destination_path,
        {dest: mimetypes.guess_type(src)[0] for dest, src in file_paths.items()},
    )
    if not response.ok:
        raise DataChainError(response.message)

    urls = response.data.get("urls", {})
    headers = response.data.get("headers", {})

    # Upload each file using the presigned URLs

    for dest_path, source_path in file_paths.items():
        if dest_path not in urls:
            raise DataChainError(f"No presigned URL found for {dest_path}")

        upload_url = urls[dest_path]["url"]
        if "fields" in urls[dest_path]:
            # S3 storage - use multipart form data upload

            # Create form data
            form_data = dict(urls[dest_path]["fields"])

            # Add Content-Type if it's required by the policy
            content_type = mimetypes.guess_type(source_path)[0]
            if content_type:
                form_data["Content-Type"] = content_type

            # Add file content
            file_content = local_fs.open(source_path, "rb").read()
            form_data["file"] = (
                os.path.basename(source_path),
                file_content,
                content_type,
            )

            # Upload using POST with form data
            upload_response = requests.post(upload_url, files=form_data, timeout=3600)
        else:
            # Read the file content
            with local_fs.open(source_path, "rb") as f:
                file_content = f.read()

            # Upload the file using the presigned URL
            upload_response = requests.request(
                response.data.get("method", "PUT"),
                upload_url,
                data=file_content,
                headers={
                    **headers,
                    "Content-Type": mimetypes.guess_type(source_path)[0],
                },
                timeout=3600,
            )

        if upload_response.status_code >= 400:
            raise DataChainError(
                f"Failed to upload {source_path} to {dest_path}. "
                f"Status: {upload_response.status_code}, "
                f"Response: {upload_response.text}"
            )

        print(f"Uploaded {source_path} to {dest_path}")

    uploads = [
        {
            "path": dst,
            "size": local_fs.info(src).get("size", 0),
        }
        for dst, src in file_paths.items()
    ]
    studio_client.save_upload_log(args.destination_path, uploads)

    print(f"Successfully uploaded {len(file_paths)} file(s)")


def _download_from_storage(args: "Namespace", local_fs: "AbstractFileSystem"):
    studio_client = _get_studio_client(args)
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

    with local_fs.open(destination_path, "wb") as f:
        f.write(requests.get(url, timeout=3600).content)

    print(f"Downloaded to {destination_path}")


def _copy_inside_storage(args: "Namespace"):
    client = _get_studio_client(args)

    response = client.copy_storage_file(
        args.source_path,
        args.destination_path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Copied {args.source_path} to {args.destination_path}")
