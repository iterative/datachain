from typing import TYPE_CHECKING

from datachain.cli.commands.storage.base import StorageImplementation
from datachain.cli.commands.storage.utils import build_file_paths, validate_upload_args

if TYPE_CHECKING:
    from datachain.client.fsspec import Client


class LocalStorageImplementation(StorageImplementation):
    def upload_to_remote(self, source_cls: "Client", destination_cls: "Client"):
        from tqdm import tqdm

        source_fs = source_cls.create_fs()
        destination_fs = destination_cls.create_fs()
        is_dir = validate_upload_args(self.args, source_fs)
        file_paths = build_file_paths(self.args, source_fs, is_dir)
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

        self.save_upload_log(None, self.args.destination_path, file_paths, source_fs)

    def download_from_remote(self, destination_cls: "Client"):
        self.catalog.cp(
            [self.args.source_path],
            self.args.destination_path,
            force=bool(self.args.force),
            update=bool(self.args.update),
            recursive=bool(self.args.recursive),
            no_glob=self.args.no_glob,
        )

    def copy_remote_to_remote(self, source_cls: "Client"):
        source_fs = source_cls.create_fs()
        source_fs.copy(
            self.args.source_path,
            self.args.destination_path,
            recursive=self.args.recursive,
        )

    def rm(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.args.path)
        fs = client_cls.create_fs()
        fs.rm(self.args.path, recursive=self.args.recursive)
        # TODO: Add storage logging.

    def mv(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.args.path)
        fs = client_cls.create_fs()
        fs.mv(self.args.path, self.args.new_path, recursive=self.args.recursive)
        # TODO: Add storage logging.
