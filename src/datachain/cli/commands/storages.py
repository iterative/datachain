import sys
from typing import TYPE_CHECKING

from datachain.error import DataChainError
from datachain.remote.storages import (
    copy_inside_storage,
    download_from_storage,
    get_studio_client,
    upload_to_storage,
)

if TYPE_CHECKING:
    from argparse import Namespace


def process_storage_command(args: "Namespace"):
    if args.cmd is None:
        print(
            f"Use 'datachain {args.command} --help' to see available options",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "rm":
        return rm_storage(args)
    if args.cmd == "mv":
        return mv_storage(args)
    if args.cmd == "cp":
        return cp_storage(args)
    raise DataChainError(f"Unknown command '{args.cmd}'.")


def rm_storage(args: "Namespace"):
    client = get_studio_client(args)

    response = client.delete_storage_file(
        args.path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Deleted {args.path}")


def mv_storage(args: "Namespace"):
    client = get_studio_client(args)

    response = client.move_storage_file(
        args.path,
        args.new_path,
        recursive=args.recursive,
    )
    if not response.ok:
        raise DataChainError(response.message)

    print(f"Moved {args.path} to {args.new_path}")


def cp_storage(args: "Namespace"):
    from datachain.client.fsspec import Client

    source_cls = Client.get_implementation(args.source_path)
    destination_cls = Client.get_implementation(args.destination_path)

    # Determine operation based on source and destination protocols
    if source_cls.protocol == "file" and destination_cls.protocol == "file":
        source_fs = source_cls.create_fs()
        source_fs.cp_file(
            args.source_path,
            args.destination_path,
        )
    elif source_cls.protocol == "file":
        source_fs = source_cls.create_fs()
        upload_to_storage(args, source_fs)
    elif destination_cls.protocol == "file":
        destination_fs = destination_cls.create_fs()
        download_from_storage(args, destination_fs)
    else:
        copy_inside_storage(args)
