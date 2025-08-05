from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.error import DataChainError

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class StorageCredentialFileHandler(CredentialBasedFileHandler):
    def upload_to_cloud(self, source_cls: "Client", destination_cls: "Client"):
        from datachain.remote.storages import upload_to_storage

        source_fs = source_cls.create_fs()
        file_paths = upload_to_storage(
            self.args.source_path,
            self.args.destination_path,
            self.args.team,
            self.args.recursive,
            source_fs,
        )
        self.save_upload_logs(self.args.destination_path, file_paths, source_fs)

    def download_from_cloud(self, destination_cls: "Client"):
        from datachain.remote.storages import download_from_storage

        destination_fs = destination_cls.create_fs()
        download_from_storage(
            self.args.source_path,
            self.args.destination_path,
            self.args.team,
            destination_fs,
        )

    def copy_cloud_to_cloud(self, source_cls: "Client"):
        from datachain.remote.storages import copy_inside_storage

        copy_inside_storage(
            self.args.source_path,
            self.args.destination_path,
            self.args.team,
            self.args.recursive,
        )

    def rm(self):
        from datachain.remote.storages import get_studio_client

        client = get_studio_client(self.args.team)
        response = client.delete_storage_file(
            self.args.path,
            recursive=self.args.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        if failed := response.data.get("failed"):
            raise DataChainError(f"Failed to remove files {'.'.join(failed.keys())}")

        print(f"Deleted {self.args.path}")

    def mv(self):
        from datachain.remote.storages import get_studio_client

        client = get_studio_client(self.args.team)
        response = client.move_storage_file(
            self.args.path,
            self.args.new_path,
            recursive=self.args.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        if failed := response.data.get("failed"):
            raise DataChainError(f"Failed to move {'.'.join(failed.keys())}")

        print(f"Moved {self.args.path} to {self.args.new_path}")
