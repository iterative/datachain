from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import StorageImplementation
from datachain.error import DataChainError

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class StudioStorageImplementation(StorageImplementation):
    def upload_to_remote(self, source_cls: "Client", destination_cls: "Client"):
        from datachain.remote.storages import upload_to_storage

        source_fs = source_cls.create_fs()
        file_paths = upload_to_storage(self.args, source_fs)
        self.save_upload_log(None, self.args.destination_path, file_paths, source_fs)

    def download_from_remote(self, destination_cls: "Client"):
        from datachain.remote.storages import download_from_storage

        destination_fs = destination_cls.create_fs()
        download_from_storage(self.args, destination_fs)

    def copy_remote_to_remote(self, source_cls: "Client"):
        from datachain.remote.storages import copy_inside_storage

        copy_inside_storage(self.args)

    def rm(self):
        from datachain.remote.storages import get_studio_client

        client = get_studio_client(self.args)

        response = client.delete_storage_file(
            self.args.path,
            recursive=self.args.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        print(f"Deleted {self.args.path}")

    def mv(self):
        from datachain.remote.storages import get_studio_client

        client = get_studio_client(self.args)

        response = client.move_storage_file(
            self.args.path,
            self.args.new_path,
            recursive=self.args.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        print(f"Moved {self.args.path} to {self.args.new_path}")
