import logging
import os
import sys
import traceback
from multiprocessing import freeze_support
from typing import Optional

from datachain.cli.utils import get_logging_level
from datachain.telemetry import telemetry

from .commands import (
    clear_cache,
    completion,
    dataset_stats,
    du,
    edit_dataset,
    garbage_collect,
    index,
    list_datasets,
    ls,
    query,
    rm_dataset,
    show,
)
from .parser import get_parser

logger = logging.getLogger("datachain")


def main(argv: Optional[list[str]] = None) -> int:
    from datachain.catalog import get_catalog

    # Required for Windows multiprocessing support
    freeze_support()

    datachain_parser = get_parser()
    args = datachain_parser.parse_args(argv)

    if args.command in ("internal-run-udf", "internal-run-udf-worker"):
        return handle_udf(args.command)

    if args.command is None:
        datachain_parser.print_help(sys.stderr)
        return 1

    logger.addHandler(logging.StreamHandler())
    logging_level = get_logging_level(args)
    logger.setLevel(logging_level)

    client_config = {
        "aws_endpoint_url": args.aws_endpoint_url,
        "anon": args.anon,
    }

    if args.debug_sql:
        # This also sets this environment variable for any subprocesses
        os.environ["DEBUG_SHOW_SQL_QUERIES"] = "True"

    error = None

    try:
        catalog = get_catalog(client_config=client_config)
        return handle_command(args, catalog, client_config)
    except BrokenPipeError as exc:
        error, return_code = handle_broken_pipe_error(exc)
        return return_code
    except (KeyboardInterrupt, Exception) as exc:
        error, return_code = handle_general_exception(exc, args, logging_level)
        return return_code
    finally:
        telemetry.send_cli_call(args.command, error=error)


def handle_command(args, catalog, client_config) -> int:
    """Handle the different CLI commands."""
    from datachain.studio import process_jobs_args, process_studio_cli_args

    command_handlers = {
        "cp": lambda: handle_cp_command(args, catalog),
        "clone": lambda: handle_clone_command(args, catalog),
        "dataset": lambda: handle_dataset_command(args, catalog),
        "ds": lambda: handle_dataset_command(args, catalog),
        "ls": lambda: handle_ls_command(args, client_config),
        "show": lambda: handle_show_command(args, catalog),
        "du": lambda: handle_du_command(args, catalog, client_config),
        "find": lambda: handle_find_command(args, catalog),
        "index": lambda: handle_index_command(args, catalog),
        "completion": lambda: handle_completion_command(args),
        "query": lambda: handle_query_command(args, catalog),
        "clear-cache": lambda: clear_cache(catalog),
        "gc": lambda: garbage_collect(catalog),
        "studio": lambda: process_studio_cli_args(args),
        "job": lambda: process_jobs_args(args),
    }

    handler = command_handlers.get(args.command)
    if handler:
        handler()
        return 0
    print(f"invalid command: {args.command}", file=sys.stderr)
    return 1


def handle_cp_command(args, catalog):
    catalog.cp(
        args.sources,
        args.output,
        force=bool(args.force),
        update=bool(args.update),
        recursive=bool(args.recursive),
        edatachain_file=None,
        edatachain_only=False,
        no_edatachain_file=True,
        no_glob=args.no_glob,
    )


def handle_clone_command(args, catalog):
    catalog.clone(
        args.sources,
        args.output,
        force=bool(args.force),
        update=bool(args.update),
        recursive=bool(args.recursive),
        no_glob=args.no_glob,
        no_cp=args.no_cp,
    )


def handle_dataset_command(args, catalog):
    if args.datasets_cmd is None:
        print(
            f"Use 'datachain {args.command} --help' to see available options",
            file=sys.stderr,
        )
        return 1

    dataset_commands = {
        "pull": lambda: catalog.pull_dataset(
            args.dataset,
            args.output,
            local_ds_name=args.local_name,
            local_ds_version=args.local_version,
            cp=args.cp,
            force=bool(args.force),
        ),
        "edit": lambda: edit_dataset(
            catalog,
            args.name,
            new_name=args.new_name,
            description=args.description,
            labels=args.labels,
            studio=args.studio,
            local=args.local,
            all=args.all,
            team=args.team,
        ),
        "ls": lambda: list_datasets(
            catalog=catalog,
            studio=args.studio,
            local=args.local,
            all=args.all,
            team=args.team,
        ),
        "rm": lambda: rm_dataset(
            catalog,
            args.name,
            version=args.version,
            force=args.force,
            studio=args.studio,
            local=args.local,
            all=args.all,
            team=args.team,
        ),
        "remove": lambda: rm_dataset(
            catalog,
            args.name,
            version=args.version,
            force=args.force,
            studio=args.studio,
            local=args.local,
            all=args.all,
            team=args.team,
        ),
        "stats": lambda: dataset_stats(
            catalog,
            args.name,
            args.version,
            show_bytes=args.bytes,
            si=args.si,
        ),
    }

    handler = dataset_commands.get(args.datasets_cmd)
    if handler:
        return handler()

    raise Exception(f"Unexpected command {args.datasets_cmd}")


def handle_ls_command(args, client_config):
    ls(
        args.sources,
        long=bool(args.long),
        studio=args.studio,
        local=args.local,
        all=args.all,
        team=args.team,
        update=bool(args.update),
        client_config=client_config,
    )


def handle_show_command(args, catalog):
    show(
        catalog,
        args.name,
        args.version,
        limit=args.limit,
        offset=args.offset,
        columns=args.columns,
        no_collapse=args.no_collapse,
        schema=args.schema,
    )


def handle_du_command(args, catalog, client_config):
    du(
        catalog,
        args.sources,
        show_bytes=args.bytes,
        depth=args.depth,
        si=args.si,
        update=bool(args.update),
        client_config=client_config,
    )


def handle_find_command(args, catalog):
    results_found = False
    for result in catalog.find(
        args.sources,
        update=bool(args.update),
        names=args.name,
        inames=args.iname,
        paths=args.path,
        ipaths=args.ipath,
        size=args.size,
        typ=args.type,
        columns=args.columns,
    ):
        print(result)
        results_found = True
    if not results_found:
        print("No results")


def handle_index_command(args, catalog):
    index(
        catalog,
        args.sources,
        update=bool(args.update),
    )


def handle_completion_command(args):
    print(completion(args.shell))


def handle_query_command(args, catalog):
    query(
        catalog,
        args.script,
        parallel=args.parallel,
        params=args.param,
    )


def handle_broken_pipe_error(exc):
    # Python flushes standard streams on exit; redirect remaining output
    # to devnull to avoid another BrokenPipeError at shutdown
    # See: https://docs.python.org/3/library/signal.html#note-on-sigpipe
    error = str(exc)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    return error, 141  # 128 + 13 (SIGPIPE)


def handle_general_exception(exc, args, logging_level):
    error = str(exc)
    if isinstance(exc, KeyboardInterrupt):
        msg = "Operation cancelled by the user"
    else:
        msg = str(exc)
    print("Error:", msg, file=sys.stderr)
    if logging_level <= logging.DEBUG:
        traceback.print_exception(
            type(exc),
            exc,
            exc.__traceback__,
            file=sys.stderr,
        )
    if args.pdb:
        import pdb  # noqa: T100

        pdb.post_mortem()
    return error, 1


def handle_udf(command):
    if command == "internal-run-udf":
        from datachain.query.dispatch import udf_entrypoint

        return udf_entrypoint()

    if command == "internal-run-udf-worker":
        from datachain.query.dispatch import udf_worker_entrypoint

        return udf_worker_entrypoint()
