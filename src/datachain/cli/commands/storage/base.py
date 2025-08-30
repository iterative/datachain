from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.client.fsspec import Client


class CredentialBasedFileHandler:
    def __init__(
        self,
        catalog: "Catalog",
        # For studio
        team: Optional[str] = None,
        # For cp
        source_path: Optional[str] = None,
        destination_path: Optional[str] = None,
        update: bool = False,
        recursive: bool = False,
        anon: bool = False,
        # For mv, rm
        path: Optional[str] = None,
        new_path: Optional[str] = None,
    ):
        self.catalog = catalog

        self.team = team
        self.source_path = source_path
        self.destination_path = destination_path
        self.update = update
        self.recursive = recursive
        self.anon = anon
        self.path = path
        self.new_path = new_path

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

        studio_client = StudioClient(team=self.team)

        uploads = [
            {
                "path": dst,
                "size": size,
            }
            for dst, size in file_paths.items()
        ]
        studio_client.save_activity_logs(destination_path, uploads)
