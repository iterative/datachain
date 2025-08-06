from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.error import DataChainError

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class StudioAuthenticatedFileHandler(CredentialBasedFileHandler):
    def cp(self):
        from datachain.client.fsspec import Client

        source_cls = Client.get_implementation(self.args.source_path)
        destination_cls = Client.get_implementation(self.args.destination_path)

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
            self.args.source_path,
            self.args.destination_path,
            recursive=self.args.recursive,
        )
        print(f"Copied {self.args.source_path} to {self.args.destination_path}")

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
        upload_info = {
            path: source_fs.info(src_path).get("size", 0)
            for path, src_path in file_paths.items()
        }
        self.save_upload_logs(self.args.destination_path, upload_info)

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
        from datachain.remote.studio import StudioClient

        client = StudioClient(team=self.args.team)
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
        from datachain.remote.studio import StudioClient

        client = StudioClient(team=self.args.team)
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
