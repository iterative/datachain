from typing import TYPE_CHECKING, Optional

from datachain.error import DataChainError

if TYPE_CHECKING:
    from argparse import Namespace

    from fsspec import AbstractFileSystem

    from datachain.catalog import Catalog
    from datachain.client.fsspec import Client
    from datachain.remote.studio import StudioClient


class StorageImplementation:
    def __init__(self, args: "Namespace", catalog: "Catalog"):
        self.args = args
        self.catalog = catalog

    def rm(self):
        raise NotImplementedError("Remove is not implemented")

    def mv(self):
        raise NotImplementedError("Move is not implemented")

    def cp(self):
        from datachain.client.fsspec import Client

        source_cls = Client.get_implementation(self.args.source_path)
        destination_cls = Client.get_implementation(self.args.destination_path)

        if source_cls.protocol == "file" and destination_cls.protocol == "file":
            self.copy_local_to_local(source_cls)
        elif source_cls.protocol == "file":
            self.upload_to_remote(source_cls, destination_cls)
        elif destination_cls.protocol == "file":
            self.download_from_remote(destination_cls)
        else:
            self.copy_remote_to_remote(source_cls)

    def copy_local_to_local(self, source_cls: "Client"):
        source_fs = source_cls.create_fs()
        source_fs.copy(
            self.args.source_path,
            self.args.destination_path,
            recursive=self.args.recursive,
        )
        print(f"Copied {self.args.source_path} to {self.args.destination_path}")

    def upload_to_remote(self, source_cls: "Client", destination_cls: "Client"):
        raise NotImplementedError("Upload to remote is not implemented")

    def download_from_remote(self, destination_cls: "Client"):
        raise NotImplementedError("Download from remote is not implemented")

    def copy_remote_to_remote(self, source_cls: "Client"):
        raise NotImplementedError("Copy remote to remote is not implemented")

    def save_upload_log(
        self,
        studio_client: Optional["StudioClient"],
        destination_path: str,
        file_paths: dict,
        local_fs: "AbstractFileSystem",
    ):
        from datachain.remote.storages import get_studio_client

        try:
            if studio_client is None:
                studio_client = get_studio_client(self.args)
        except DataChainError:
            return

        uploads = [
            {
                "path": dst,
                "size": local_fs.info(src).get("size", 0),
            }
            for dst, src in file_paths.items()
        ]
        studio_client.save_upload_log(destination_path, uploads)
