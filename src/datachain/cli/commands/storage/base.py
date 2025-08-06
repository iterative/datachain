from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

    from datachain.catalog import Catalog
    from datachain.client.fsspec import Client


class CredentialBasedFileHandler:
    def __init__(self, args: "Namespace", catalog: "Catalog"):
        self.args = args
        self.catalog = catalog

    def rm(self):
        raise NotImplementedError("Remove is not implemented")

    def mv(self):
        raise NotImplementedError("Move is not implemented")

    def cp(self):
        raise NotImplementedError("Copy is not implemented")

    def upload_to_cloud(self, source_cls: "Client", destination_cls: "Client"):
        raise NotImplementedError("Upload to remote is not implemented")

    def download_from_cloud(self, destination_cls: "Client"):
        raise NotImplementedError("Download from remote is not implemented")

    def copy_cloud_to_cloud(self, source_cls: "Client"):
        raise NotImplementedError("Copy remote to remote is not implemented")

    def save_upload_logs(
        self,
        destination_path: str,
        file_paths: dict,  # {destination_path: size}
    ):
        from datachain.remote.studio import StudioClient, is_token_set

        if not is_token_set():
            return

        studio_client = StudioClient(team=self.args.team)

        uploads = [
            {
                "path": dst,
                "size": size,
            }
            for dst, size in file_paths.items()
        ]
        studio_client.save_activity_logs(destination_path, uploads)
