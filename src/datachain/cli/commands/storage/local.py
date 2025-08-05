from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.cli.commands.storage.utils import build_file_paths, validate_upload_args
from datachain.error import DataChainError

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class LocalCredentialsBasedFileHandler(CredentialBasedFileHandler):
    def upload_to_cloud(self, source_cls: "Client", destination_cls: "Client"):
        from tqdm import tqdm

        source_fs = source_cls.create_fs()
        destination_fs = destination_cls.create_fs()
        is_dir = validate_upload_args(
            self.args.source_path, self.args.recursive, source_fs
        )
        file_paths = build_file_paths(
            self.args.source_path, self.args.destination_path, source_fs, is_dir
        )
        destination_bucket, _ = destination_cls.parse_url(self.args.destination_path)

        for dest_path, source_path in file_paths.items():
            file_size = source_fs.info(source_path)["size"]
            print(f"Uploading {source_path} to {dest_path}")

            with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                with source_fs.open(source_path, "rb") as source_file:
                    with destination_fs.open(
                        f"{destination_bucket}/{dest_path}", "wb"
                    ) as dest_file:
                        while True:
                            chunk = source_file.read(8192)
                            if not chunk:
                                break
                            dest_file.write(chunk)
                            pbar.update(len(chunk))

        self.save_upload_logs(self.args.destination_path, file_paths, source_fs)

    def download_from_cloud(self, destination_cls: "Client"):
        self.catalog.cp(
            [self.args.source_path],
            self.args.destination_path,
            force=bool(self.args.force),
            update=bool(self.args.update),
            recursive=bool(self.args.recursive),
            no_glob=self.args.no_glob,
        )
        print(f"Downloaded {self.args.source_path} to {self.args.destination_path}")

    def copy_cloud_to_cloud(self, source_cls: "Client"):
        source_fs = source_cls.create_fs()
        source_fs.copy(
            self.args.source_path,
            self.args.destination_path,
            recursive=self.args.recursive,
        )

        _, dest = source_cls.split_url(self.args.destination_path)
        file_paths = {dest: self.args.source_path}
        print(f"Copied {self.args.source_path} to {self.args.destination_path}")
        self.save_upload_logs(self.args.destination_path, file_paths, source_fs)

    def rm(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.args.path)
        fs = client_cls.create_fs()
        fs.rm(self.args.path, recursive=self.args.recursive)
        if client_cls.protocol != "file":
            _, path = client_cls.split_url(self.args.path)
            self.save_deleted_logs(self.args.path, [path])

        print(f"Deleted {self.args.path}")

    def mv(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.args.path)
        fs = client_cls.create_fs()
        size = fs.info(self.args.path).get("size", 0)
        fs.mv(self.args.path, self.args.new_path, recursive=self.args.recursive)

        if client_cls.protocol != "file":
            _, src = client_cls.split_url(self.args.path)
            _, dst = client_cls.split_url(self.args.new_path)
            self.save_moved_logs(self.args.new_path, {src: (dst, size)})
        print(f"Moved {self.args.path} to {self.args.new_path}")

    def save_deleted_logs(
        self,
        destination_path: str,
        file_paths: list[str],  # {destination_path: source}
    ):
        from datachain.remote.storages import get_studio_client

        try:
            studio_client = get_studio_client(self.args.team)
        except DataChainError:
            return

        studio_client.save_activity_logs(
            destination_path,
            deleted_paths=file_paths,
        )

    def save_moved_logs(
        self,
        destination_path: str,
        file_paths: dict[str, tuple[str, int]],  # {old_path: (new_path, size)},
    ):
        from datachain.remote.storages import get_studio_client

        try:
            studio_client = get_studio_client(self.args.team)
        except DataChainError:
            return

        moved_paths = [(dst, src, size) for dst, (src, size) in file_paths.items()]

        studio_client.save_activity_logs(
            destination_path,
            moved_paths=moved_paths,
        )
