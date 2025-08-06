from pathlib import Path

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.error import DataChainError
from datachain.lib.dc.storage import read_storage


class LocalCredentialsBasedFileHandler(CredentialBasedFileHandler):
    def cp(self):
        from datachain.client.fsspec import Client

        source_path = self.args.source_path
        destination_path = self.args.destination_path
        destination_cls = Client.get_implementation(destination_path)

        source_cls = Client.get_implementation(source_path)
        source_fs = source_cls.create_fs()
        _, relative_to = source_cls.split_url(source_path)

        update = self.args.update
        if source_cls.protocol == "file":
            update = True
            relative_to = None

        info = source_fs.info(source_path)
        is_file = info["type"] == "file"

        if info["type"] == "directory" and not self.args.recursive:
            raise ValueError("Source is a directory, but recursive is not specified")

        chain = read_storage(source_path, update=update)
        file_paths = {}

        if not is_file:
            chain.to_storage(
                destination_path, placement="normpath", relative_to=relative_to
            )

        for (file,) in chain.to_iter("file"):
            if is_file:
                dst = (
                    Path(destination_path) / file.name
                    if destination_path.endswith("/")
                    else destination_path
                )
                file.save(dst)
            else:
                dst = file.get_destination_path(
                    destination_path, "normpath", relative_to=relative_to
                )
            _, dst_path = destination_cls.split_url(str(dst))
            file_paths[dst_path] = file.size

        if destination_cls.protocol != "file":
            self.save_upload_logs(destination_path, file_paths)
        print(f"Copied {source_path} to {destination_path}")

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
