from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.error import DataChainError

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class StudioAuthenticatedFileHandler(CredentialBasedFileHandler):
    def cp(self):
        from datachain.client.fsspec import Client

        source_cls = Client.get_implementation(self.source_path)
        destination_cls = Client.get_implementation(self.destination_path)

        if source_cls.protocol == "file" and destination_cls.protocol == "file":
            self.copy_local_to_local(source_cls)
        elif source_cls.protocol == "file":
            self.upload_to_cloud(source_cls, destination_cls)
        elif destination_cls.protocol == "file":
            self.download_from_cloud(destination_cls)
        elif source_cls.protocol == destination_cls.protocol:
            self.copy_cloud_to_cloud(source_cls)
        else:
            raise DataChainError("Cannot copy between different protocols yet")

    def copy_local_to_local(self, source_cls: "Client"):
        source_fs = source_cls.create_fs()
        source_fs.copy(
            self.source_path,
            self.destination_path,
            recursive=self.recursive,
        )
        print(f"Copied {self.source_path} to {self.destination_path}")

    def upload_to_cloud(self, source_cls: "Client", destination_cls: "Client"):
        from datachain.remote.storages import upload_to_storage

        assert self.source_path and self.destination_path, (
            "Source and destination paths are required"
        )

        source_fs = source_cls.create_fs()
        file_paths = upload_to_storage(
            self.source_path,
            self.destination_path,
            self.team,
            self.recursive,
            source_fs,
        )
        upload_info = {
            path: source_fs.info(src_path).get("size", 0)
            for path, src_path in file_paths.items()
        }
        self.save_upload_logs(self.destination_path, upload_info)

    def download_from_cloud(self, destination_cls: "Client"):
        from datachain.remote.storages import download_from_storage

        assert self.source_path and self.destination_path, (
            "Source and destination paths are required"
        )

        destination_fs = destination_cls.create_fs()
        download_from_storage(
            self.source_path,
            self.destination_path,
            self.team,
            destination_fs,
        )

    def copy_cloud_to_cloud(self, source_cls: "Client"):
        from datachain.remote.storages import copy_inside_storage

        assert self.source_path and self.destination_path, (
            "Source and destination paths are required"
        )

        copy_inside_storage(
            self.source_path,
            self.destination_path,
            self.team,
            self.recursive,
        )

    def rm(self):
        from datachain.remote.studio import StudioClient

        assert self.path, "Path is required"

        client = StudioClient(team=self.team)
        response = client.delete_storage_file(
            self.path,
            recursive=self.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        if failed := response.data.get("failed"):
            raise DataChainError(f"Failed to remove files {'.'.join(failed.keys())}")

        print(f"Deleted {self.path}")

    def mv(self):
        from datachain.remote.studio import StudioClient

        client = StudioClient(team=self.team)
        response = client.move_storage_file(
            self.path,
            self.new_path,
            recursive=self.recursive,
        )
        if not response.ok:
            raise DataChainError(response.message)

        if failed := response.data.get("failed"):
            raise DataChainError(f"Failed to move {'.'.join(failed.keys())}")

        print(f"Moved {self.path} to {self.new_path}")
