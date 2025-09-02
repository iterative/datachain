from pathlib import Path

from datachain.cli.commands.storage.base import CredentialBasedFileHandler
from datachain.lib.dc.storage import read_storage


class LocalCredentialsBasedFileHandler(CredentialBasedFileHandler):
    def cp(self):
        from datachain.client.fsspec import Client

        source_path = self.source_path
        destination_path = self.destination_path
        destination_cls = Client.get_implementation(destination_path)

        source_cls = Client.get_implementation(source_path)
        source_fs = source_cls.create_fs()
        _, relative_to = source_cls.split_url(source_path)

        update = self.update
        if source_cls.protocol == "file":
            update = True
            relative_to = None

        is_file = source_fs.isfile(source_path)
        if not self.recursive and source_fs.isdir(source_path):
            raise ValueError("Source is a directory, but recursive is not specified")

        chain = read_storage(source_path, update=update, anon=self.anon).settings(
            cache=True, parallel=10
        )
        file_paths = {}

        def _calculate_full_dst(file) -> str:
            if is_file:
                return (
                    str(Path(destination_path) / file.name)
                    if destination_path.endswith("/")
                    else destination_path
                )
            return file.get_destination_path(
                destination_path, "normpath", relative_to=relative_to
            )

        chain = chain.map(full_dst=_calculate_full_dst)
        if is_file:
            chain = chain.map(save_file=lambda file, full_dst: str(file.save(full_dst)))
        else:
            chain.to_storage(
                destination_path, placement="normpath", relative_to=relative_to
            )

        chain = chain.map(
            dst_path=lambda full_dst: destination_cls.split_url(full_dst)[1]
        )
        file_paths = dict(chain.to_list("dst_path", "file.size"))

        if destination_cls.protocol != "file":
            self.save_upload_logs(destination_path, file_paths)
        print(f"Copied {source_path} to {destination_path}")

    def rm(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.path)
        fs = client_cls.create_fs()
        fs.rm(self.path, recursive=self.recursive)
        if client_cls.protocol != "file":
            _, path = client_cls.split_url(self.path)
            self.save_deleted_logs(self.path, [path])

        print(f"Deleted {self.path}")

    def mv(self):
        from datachain.client.fsspec import Client

        client_cls = Client.get_implementation(self.path)
        fs = client_cls.create_fs()
        size = fs.info(self.path).get("size", 0)
        fs.mv(self.path, self.new_path, recursive=self.recursive)

        if client_cls.protocol != "file":
            _, src = client_cls.split_url(self.path)
            _, dst = client_cls.split_url(self.new_path)
            self.save_moved_logs(self.new_path, {src: (dst, size)})
        print(f"Moved {self.path} to {self.new_path}")

    def save_deleted_logs(
        self,
        destination_path: str,
        file_paths: list[str],  # {destination_path: source}
    ):
        from datachain.remote.studio import StudioClient, is_token_set

        if not is_token_set():
            return

        studio_client = StudioClient(team=self.team)

        studio_client.save_activity_logs(
            destination_path,
            deleted_paths=file_paths,
        )

    def save_moved_logs(
        self,
        destination_path: str,
        file_paths: dict[str, tuple[str, int]],  # {old_path: (new_path, size)},
    ):
        from datachain.remote.studio import StudioClient, is_token_set

        if not is_token_set():
            return
        studio_client = StudioClient(team=self.team)

        moved_paths = [(dst, src, size) for dst, (src, size) in file_paths.items()]

        studio_client.save_activity_logs(
            destination_path,
            moved_paths=moved_paths,
        )
