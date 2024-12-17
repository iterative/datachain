import logging
import os
import shlex
import sys
import traceback
from argparse import Action, ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Iterable, Iterator, Sequence
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
from multiprocessing import freeze_support
from typing import TYPE_CHECKING, Optional, Union

import shtab
from tabulate import tabulate

from datachain import Session, utils
from datachain.cli_utils import BooleanOptionalAction, CommaSeparatedArgs, KeyValueArgs
from datachain.config import Config
from datachain.error import DataChainError, DatasetNotFoundError
from datachain.lib.dc import DataChain
from datachain.studio import (
    edit_studio_dataset,
    list_datasets,
    process_studio_cli_args,
    remove_studio_dataset,
)
from datachain.telemetry import telemetry

if TYPE_CHECKING:
    from datachain.catalog import Catalog

logger = logging.getLogger("datachain")

TTL_HUMAN = "4h"
TTL_INT = 4 * 60 * 60
FIND_COLUMNS = ["du", "name", "path", "size", "type"]


def human_time_type(value_str: str, can_be_none: bool = False) -> Optional[int]:
    value = utils.human_time_to_int(value_str)

    if value:
        return value
    if can_be_none:
        return None

    raise ArgumentTypeError(
        "This option supports only a human-readable time interval like 12h or 4w."
    )


def parse_find_column(column: str) -> str:
    column_lower = column.strip().lower()
    if column_lower in FIND_COLUMNS:
        return column_lower
    raise ArgumentTypeError(
        f"Invalid column for find: '{column}' Options are: {','.join(FIND_COLUMNS)}"
    )


def find_columns_type(
    columns_str: str,
    default_colums_str: str = "path",
) -> list[str]:
    if not columns_str:
        columns_str = default_colums_str

    return [parse_find_column(c) for c in columns_str.split(",")]


def add_sources_arg(parser: ArgumentParser, nargs: Union[str, int] = "+") -> Action:
    return parser.add_argument(
        "sources",
        type=str,
        nargs=nargs,
        help="Data sources - paths to cloud storage dirs",
    )


def add_show_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        action="store",
        default=10,
        type=int,
        help="Number of rows to show",
    )
    parser.add_argument(
        "--offset",
        action="store",
        default=0,
        type=int,
        help="Number of rows to offset",
    )
    parser.add_argument(
        "--columns",
        default=[],
        action=CommaSeparatedArgs,
        help="Columns to show",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        default=False,
        help="Do not collapse the columns",
    )


def add_studio_parser(subparsers, parent_parser) -> None:
    studio_help = "Commands to authenticate DataChain with Iterative Studio"
    studio_description = (
        "Authenticate DataChain with Studio and set the token. "
        "Once this token has been properly configured,\n"
        "DataChain will utilize it for seamlessly sharing datasets\n"
        "and using Studio features from CLI"
    )

    studio_parser = subparsers.add_parser(
        "studio",
        parents=[parent_parser],
        description=studio_description,
        help=studio_help,
    )
    studio_subparser = studio_parser.add_subparsers(
        dest="cmd",
        help="Use `DataChain studio CMD --help` to display command-specific help.",
        required=True,
    )

    studio_login_help = "Authenticate DataChain with Studio host"
    studio_login_description = (
        "By default, this command authenticates the DataChain with Studio\n"
        "using default scopes and assigns a random name as the token name."
    )
    login_parser = studio_subparser.add_parser(
        "login",
        parents=[parent_parser],
        description=studio_login_description,
        help=studio_login_help,
    )

    login_parser.add_argument(
        "-H",
        "--hostname",
        action="store",
        default=None,
        help="The hostname of the Studio instance to authenticate with.",
    )
    login_parser.add_argument(
        "-s",
        "--scopes",
        action="store",
        default=None,
        help="The scopes for the authentication token. ",
    )

    login_parser.add_argument(
        "-n",
        "--name",
        action="store",
        default=None,
        help="The name of the authentication token. It will be used to\n"
        "identify token shown in Studio profile.",
    )

    login_parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Use authentication flow based on user code.\n"
        "You will be presented with user code to enter in browser.\n"
        "DataChain will also use this if it cannot launch browser on your behalf.",
    )

    studio_logout_help = "Logout user from Studio"
    studio_logout_description = "This removes the studio token from your global config."

    studio_subparser.add_parser(
        "logout",
        parents=[parent_parser],
        description=studio_logout_description,
        help=studio_logout_help,
    )

    studio_team_help = "Set the default team for DataChain"
    studio_team_description = (
        "Set the default team for DataChain to use when interacting with Studio."
    )

    team_parser = studio_subparser.add_parser(
        "team",
        parents=[parent_parser],
        description=studio_team_description,
        help=studio_team_help,
    )
    team_parser.add_argument(
        "team_name",
        action="store",
        help="The name of the team to set as the default.",
    )
    team_parser.add_argument(
        "--global",
        action="store_true",
        default=False,
        help="Set the team globally for all DataChain projects.",
    )

    studio_token_help = "View the token datachain uses to contact Studio"  # noqa: S105 # nosec B105

    studio_subparser.add_parser(
        "token",
        parents=[parent_parser],
        description=studio_token_help,
        help=studio_token_help,
    )

    studio_ls_dataset_help = "List the available datasets from Studio"
    studio_ls_dataset_description = (
        "This command lists all the datasets available in Studio.\n"
        "It will show the dataset name and the number of versions available."
    )

    ls_dataset_parser = studio_subparser.add_parser(
        "datasets",
        parents=[parent_parser],
        description=studio_ls_dataset_description,
        help=studio_ls_dataset_help,
    )
    ls_dataset_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to list datasets for. By default, it will use team from config.",
    )

    studio_run_help = "Run a job in Studio"
    studio_run_description = "This command runs a job in Studio."

    studio_run_parser = studio_subparser.add_parser(
        "run",
        parents=[parent_parser],
        description=studio_run_description,
        help=studio_run_help,
    )

    studio_run_parser.add_argument(
        "query_file",
        action="store",
        help="The query file to run.",
    )

    studio_run_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to run a job for. By default, it will use team from config.",
    )
    studio_run_parser.add_argument(
        "--env-file",
        action="store",
        help="File containing environment variables to set for the job.",
    )

    studio_run_parser.add_argument(
        "--env",
        nargs="+",
        help="Environment variable. Can be specified multiple times. Format: KEY=VALUE",
    )

    studio_run_parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use for the job.",
    )
    studio_run_parser.add_argument(
        "--files",
        nargs="+",
        help="Files to include in the job.",
    )
    studio_run_parser.add_argument(
        "--python-version",
        action="store",
        help="Python version to use for the job (e.g. '3.9', '3.10', '3.11').",
    )
    studio_run_parser.add_argument(
        "--req-file",
        action="store",
        help="File containing Python package requirements.",
    )

    studio_run_parser.add_argument(
        "--req",
        nargs="+",
        help="Python package requirement. Can be specified multiple times.",
    )

    studio_cancel_help = "Cancel a job in Studio"
    studio_cancel_description = "This command cancels a job in Studio."

    studio_cancel_parser = studio_subparser.add_parser(
        "cancel",
        parents=[parent_parser],
        description=studio_cancel_description,
        help=studio_cancel_help,
    )

    studio_cancel_parser.add_argument(
        "job_id",
        action="store",
        help="The job ID to cancel.",
    )
    studio_cancel_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to cancel a job for. By default, it will use team from config.",
    )


def get_parser() -> ArgumentParser:  # noqa: PLR0915
    try:
        __version__ = version("datachain")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

    parser = ArgumentParser(
        description="DataChain: Wrangle unstructured AI data at scale", prog="datachain"
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--aws-endpoint-url",
        type=str,
        help="AWS endpoint URL",
    )
    parent_parser.add_argument(
        "--anon",
        action="store_true",
        help="AWS anon (aka awscli's --no-sign-request)",
    )
    parent_parser.add_argument(
        "-u", "--update", action="count", default=0, help="Update cache"
    )
    parent_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbose"
    )
    parent_parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Be quiet"
    )
    parent_parser.add_argument(
        "--debug-sql",
        action="store_true",
        default=False,
        help="Show All SQL Queries (very verbose output, for debugging only)",
    )
    parent_parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Drop into the pdb debugger on fatal exception",
    )

    subp = parser.add_subparsers(
        title="Available Commands",
        metavar="command",
        dest="command",
        help=f"Use `{parser.prog} command --help` for command-specific help.",
        required=True,
    )
    parse_cp = subp.add_parser(
        "cp", parents=[parent_parser], description="Copy data files from the cloud"
    )
    add_sources_arg(parse_cp).complete = shtab.DIR  # type: ignore[attr-defined]
    parse_cp.add_argument("output", type=str, help="Output")
    parse_cp.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating outputs",
    )
    parse_cp.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_cp.add_argument(
        "--no-glob",
        default=False,
        action="store_true",
        help="Do not expand globs (such as * or ?)",
    )

    parse_clone = subp.add_parser(
        "clone", parents=[parent_parser], description="Copy data files from the cloud"
    )
    add_sources_arg(parse_clone).complete = shtab.DIR  # type: ignore[attr-defined]
    parse_clone.add_argument("output", type=str, help="Output")
    parse_clone.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating outputs",
    )
    parse_clone.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_clone.add_argument(
        "--no-glob",
        default=False,
        action="store_true",
        help="Do not expand globs (such as * or ?)",
    )
    parse_clone.add_argument(
        "--no-cp",
        default=False,
        action="store_true",
        help="Do not copy files, just create a dataset",
    )
    parse_clone.add_argument(
        "--edatachain",
        default=False,
        action="store_true",
        help="Create a .edatachain file",
    )
    parse_clone.add_argument(
        "--edatachain-file",
        help="Use a different filename for the resulting .edatachain file",
    )

    add_studio_parser(subp, parent_parser)

    datasets_parser = subp.add_parser(
        "datasets",
        aliases=["ds"],
        parents=[parent_parser],
        description="Commands for managing datasers",
    )
    datasets_subparser = datasets_parser.add_subparsers(
        dest="datasets_cmd",
        help="Use `datachain datasets CMD --help` to display command specific help",
    )

    parse_pull = datasets_subparser.add_parser(
        "pull",
        parents=[parent_parser],
        description="Pull specific dataset version from SaaS",
    )
    parse_pull.add_argument(
        "dataset",
        type=str,
        help="Name and version of remote dataset created in SaaS",
    )
    parse_pull.add_argument("-o", "--output", type=str, help="Output")
    parse_pull.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Force creating outputs",
    )
    parse_pull.add_argument(
        "-r",
        "-R",
        "--recursive",
        default=False,
        action="store_true",
        help="Copy directories recursively",
    )
    parse_pull.add_argument(
        "--cp",
        default=False,
        action="store_true",
        help="Copy actual files after pulling remote dataset into local DB",
    )
    parse_pull.add_argument(
        "--edatachain",
        default=False,
        action="store_true",
        help="Create .edatachain file",
    )
    parse_pull.add_argument(
        "--edatachain-file",
        help="Use a different filename for the resulting .edatachain file",
    )
    parse_pull.add_argument(
        "--local-name",
        action="store",
        default=None,
        help="Name of the local dataset",
    )
    parse_pull.add_argument(
        "--local-version",
        action="store",
        default=None,
        help="Version of the local dataset",
    )

    parse_edit_dataset = datasets_subparser.add_parser(
        "edit", parents=[parent_parser], description="Edit dataset metadata"
    )
    parse_edit_dataset.add_argument("name", type=str, help="Dataset name")
    parse_edit_dataset.add_argument(
        "--new-name",
        action="store",
        help="Dataset new name",
    )
    parse_edit_dataset.add_argument(
        "--description",
        action="store",
        help="Dataset description",
    )
    parse_edit_dataset.add_argument(
        "--labels",
        nargs="+",
        help="Dataset labels",
    )
    parse_edit_dataset.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="Edit dataset from Studio",
    )
    parse_edit_dataset.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="Edit local dataset only",
    )
    parse_edit_dataset.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="Edit both datasets from studio and local",
    )
    parse_edit_dataset.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to edit a dataset. By default, it will use team from config.",
    )

    datasets_parser = datasets_subparser.add_parser(
        "ls", parents=[parent_parser], description="List datasets"
    )
    datasets_parser.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="List the files in the Studio",
    )
    datasets_parser.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="List local files only",
    )
    datasets_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="List all files including hidden files",
    )
    datasets_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to list datasets for. By default, it will use team from config.",
    )

    rm_dataset_parser = datasets_subparser.add_parser(
        "rm", parents=[parent_parser], description="Removes dataset", aliases=["remove"]
    )
    rm_dataset_parser.add_argument("name", type=str, help="Dataset name")
    rm_dataset_parser.add_argument(
        "--version",
        action="store",
        default=None,
        type=int,
        help="Dataset version",
    )
    rm_dataset_parser.add_argument(
        "--force",
        default=False,
        action=BooleanOptionalAction,
        help="Force delete registered dataset with all of it's versions",
    )
    rm_dataset_parser.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="Remove dataset from Studio",
    )
    rm_dataset_parser.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="Remove local datasets only",
    )
    rm_dataset_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="Remove both local and studio",
    )
    rm_dataset_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to delete a dataset. By default, it will use team from config.",
    )

    dataset_stats_parser = datasets_subparser.add_parser(
        "stats",
        parents=[parent_parser],
        description="Shows basic dataset stats",
    )
    dataset_stats_parser.add_argument("name", type=str, help="Dataset name")
    dataset_stats_parser.add_argument(
        "--version",
        action="store",
        default=None,
        type=int,
        help="Dataset version",
    )
    dataset_stats_parser.add_argument(
        "-b",
        "--bytes",
        default=False,
        action="store_true",
        help="Display size in bytes instead of human-readable size",
    )
    dataset_stats_parser.add_argument(
        "--si",
        default=False,
        action="store_true",
        help="Display size using powers of 1000 not 1024",
    )

    parse_ls = subp.add_parser(
        "ls", parents=[parent_parser], description="List storage contents"
    )
    add_sources_arg(parse_ls, nargs="*")
    parse_ls.add_argument(
        "-l",
        "--long",
        action="count",
        default=0,
        help="List files in the long format",
    )
    parse_ls.add_argument(
        "--studio",
        action="store_true",
        default=False,
        help="List the files in the Studio",
    )
    parse_ls.add_argument(
        "-L",
        "--local",
        action="store_true",
        default=False,
        help="List local files only",
    )
    parse_ls.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=True,
        help="List all files including hidden files",
    )
    parse_ls.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to list datasets for. By default, it will use team from config.",
    )

    parse_du = subp.add_parser(
        "du", parents=[parent_parser], description="Display space usage"
    )
    add_sources_arg(parse_du)
    parse_du.add_argument(
        "-b",
        "--bytes",
        default=False,
        action="store_true",
        help="Display sizes in bytes instead of human-readable sizes",
    )
    parse_du.add_argument(
        "-d",
        "--depth",
        "--max-depth",
        default=0,
        type=int,
        metavar="N",
        help=(
            "Display sizes for N directory depths below the given directory, "
            "the default is 0 (summarize provided directory only)."
        ),
    )
    parse_du.add_argument(
        "--si",
        default=False,
        action="store_true",
        help="Display sizes using powers of 1000 not 1024",
    )

    parse_find = subp.add_parser(
        "find", parents=[parent_parser], description="Search in a directory hierarchy"
    )
    add_sources_arg(parse_find)
    parse_find.add_argument(
        "--name",
        type=str,
        action="append",
        help="Filename to match pattern.",
    )
    parse_find.add_argument(
        "--iname",
        type=str,
        action="append",
        help="Like -name but case insensitive.",
    )
    parse_find.add_argument(
        "--path",
        type=str,
        action="append",
        help="Path to match pattern.",
    )
    parse_find.add_argument(
        "--ipath",
        type=str,
        action="append",
        help="Like -path but case insensitive.",
    )
    parse_find.add_argument(
        "--size",
        type=str,
        help=(
            "Filter by size (+ is greater or equal, - is less or equal). "
            "Specified size is in bytes, or use a suffix like K, M, G for "
            "kilobytes, megabytes, gigabytes, etc."
        ),
    )
    parse_find.add_argument(
        "--type",
        type=str,
        help='File type: "f" - regular, "d" - directory',
    )
    parse_find.add_argument(
        "-c",
        "--columns",
        type=find_columns_type,
        default=None,
        help=(
            "A comma-separated list of columns to print for each result. "
            f"Options are: {','.join(FIND_COLUMNS)} (Default: path)"
        ),
    )

    parse_index = subp.add_parser(
        "index", parents=[parent_parser], description="Index storage location"
    )
    add_sources_arg(parse_index)

    show_parser = subp.add_parser(
        "show",
        parents=[parent_parser],
        description="Create a new dataset with a query script",
    )
    show_parser.add_argument("name", type=str, help="Dataset name")
    show_parser.add_argument(
        "--version",
        action="store",
        default=None,
        type=int,
        help="Dataset version",
    )
    show_parser.add_argument("--schema", action="store_true", help="Show schema")
    add_show_args(show_parser)

    query_parser = subp.add_parser(
        "query",
        parents=[parent_parser],
        description="Create a new dataset with a query script",
    )
    query_parser.add_argument(
        "script", metavar="<script.py>", type=str, help="Filepath for script"
    )
    query_parser.add_argument(
        "--parallel",
        nargs="?",
        type=int,
        const=-1,
        default=None,
        metavar="N",
        help=(
            "Use multiprocessing to run any query script UDFs with N worker processes. "
            "N defaults to the CPU count."
        ),
    )
    query_parser.add_argument(
        "-p",
        "--param",
        metavar="param=value",
        nargs=1,
        action=KeyValueArgs,
        help="Query parameters",
    )

    subp.add_parser(
        "clear-cache", parents=[parent_parser], description="Clear the local file cache"
    )
    subp.add_parser(
        "gc", parents=[parent_parser], description="Garbage collect temporary tables"
    )

    subp.add_parser("internal-run-udf", parents=[parent_parser])
    subp.add_parser("internal-run-udf-worker", parents=[parent_parser])
    add_completion_parser(subp, [parent_parser])
    return parser


def add_completion_parser(subparsers, parents):
    parser = subparsers.add_parser(
        "completion",
        parents=parents,
        description="Output shell completion script",
    )
    parser.add_argument(
        "-s",
        "--shell",
        help="Shell syntax for completions.",
        default="bash",
        choices=shtab.SUPPORTED_SHELLS,
    )


def get_logging_level(args: Namespace) -> int:
    if args.quiet:
        return logging.CRITICAL
    if args.verbose:
        return logging.DEBUG
    return logging.INFO


def ls_urls(
    sources,
    catalog: "Catalog",
    long: bool = False,
    **kwargs,
) -> Iterator[tuple[str, Iterator[str]]]:
    curr_dir = None
    value_iterables = []
    for next_dir, values in _ls_urls_flat(sources, long, catalog, **kwargs):
        if curr_dir is None or next_dir == curr_dir:  # type: ignore[unreachable]
            value_iterables.append(values)
        else:
            yield curr_dir, chain(*value_iterables)  # type: ignore[unreachable]
            value_iterables = [values]
        curr_dir = next_dir
    if curr_dir is not None:
        yield curr_dir, chain(*value_iterables)


def _node_data_to_ls_values(row, long_format=False):
    from datachain.node import DirType, long_line_str

    name = row[0]
    is_dir = row[1] == DirType.DIR
    ending = "/" if is_dir else ""
    value = name + ending
    if long_format:
        last_modified = row[2]
        timestamp = last_modified if not is_dir else None
        return long_line_str(value, timestamp)
    return value


def _ls_urls_flat(
    sources,
    long: bool,
    catalog: "Catalog",
    **kwargs,
) -> Iterator[tuple[str, Iterator[str]]]:
    from datachain.client import Client
    from datachain.node import long_line_str

    for source in sources:
        client_cls = Client.get_implementation(source)
        if client_cls.is_root_url(source):
            buckets = client_cls.ls_buckets(**catalog.client_config)
            if long:
                values = (long_line_str(b.name, b.created) for b in buckets)
            else:
                values = (b.name for b in buckets)
            yield source, values
        else:
            found = False
            fields = ["name", "dir_type"]
            if long:
                fields.append("last_modified")
            for data_source, results in catalog.ls([source], fields=fields, **kwargs):
                values = (_node_data_to_ls_values(r, long) for r in results)
                found = True
                yield data_source.dirname(), values
            if not found:
                raise FileNotFoundError(f"No such file or directory: {source}")


def ls_local(
    sources,
    long: bool = False,
    catalog: Optional["Catalog"] = None,
    client_config=None,
    **kwargs,
):
    if catalog is None:
        from .catalog import get_catalog

        catalog = get_catalog(client_config=client_config)
    if sources:
        actual_sources = list(ls_urls(sources, catalog=catalog, long=long, **kwargs))
        if len(actual_sources) == 1:
            for _, entries in actual_sources:
                for entry in entries:
                    print(format_ls_entry(entry))
        else:
            first = True
            for source, entries in actual_sources:
                # print a newline between directory listings
                if first:
                    first = False
                else:
                    print()
                if source:
                    print(f"{source}:")
                for entry in entries:
                    print(format_ls_entry(entry))
    else:
        chain = DataChain.listings()
        for ls in chain.collect("listing"):
            print(format_ls_entry(f"{ls.uri}@v{ls.version}"))  # type: ignore[union-attr]


def format_ls_entry(entry: str) -> str:
    if entry.endswith("/") or not entry:
        entry = shlex.quote(entry[:-1])
        return f"{entry}/"
    return shlex.quote(entry)


def ls_remote(
    paths: Iterable[str],
    long: bool = False,
    team: Optional[str] = None,
):
    from datachain.node import long_line_str
    from datachain.remote.studio import StudioClient

    client = StudioClient(team=team)
    first = True
    for path, response in client.ls(paths):
        if not first:
            print()
        if not response.ok or response.data is None:
            print(f"{path}:\n  Error: {response.message}\n")
            continue

        print(f"{path}:")
        if long:
            for row in response.data:
                entry = long_line_str(
                    row["name"] + ("/" if row["dir_type"] else ""),
                    row["last_modified"],
                )
                print(format_ls_entry(entry))
        else:
            for row in response.data:
                entry = row["name"] + ("/" if row["dir_type"] else "")
                print(format_ls_entry(entry))
        first = False


def ls(
    sources,
    long: bool = False,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
    **kwargs,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = _determine_flavors(studio, local, all, token)

    if all or local:
        ls_local(sources, long=long, **kwargs)

    if (all or studio) and token:
        ls_remote(sources, long=long, team=team)


def datasets(
    catalog: "Catalog",
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = _determine_flavors(studio, local, all, token)

    local_datasets = set(list_datasets_local(catalog)) if all or local else set()
    studio_datasets = (
        set(list_datasets(team=team)) if (all or studio) and token else set()
    )

    rows = [
        _datasets_tabulate_row(
            name=name,
            version=version,
            both=(all or (local and studio)) and token,
            local=(name, version) in local_datasets,
            studio=(name, version) in studio_datasets,
        )
        for name, version in local_datasets.union(studio_datasets)
    ]

    print(tabulate(rows, headers="keys"))


def list_datasets_local(catalog: "Catalog"):
    for d in catalog.ls_datasets():
        for v in d.versions:
            yield (d.name, v.version)


def _datasets_tabulate_row(name, version, both, local, studio):
    row = {
        "Name": name,
        "Version": version,
    }
    if both:
        row["Studio"] = "\u2714" if studio else "\u2716"
        row["Local"] = "\u2714" if local else "\u2716"
    return row


def rm_dataset(
    catalog: "Catalog",
    name: str,
    version: Optional[int] = None,
    force: Optional[bool] = False,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = _determine_flavors(studio, local, all, token)

    if all or local:
        try:
            catalog.remove_dataset(name, version=version, force=force)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)

    if (all or studio) and token:
        remove_studio_dataset(team, name, version, force)


def edit_dataset(
    catalog: "Catalog",
    name: str,
    new_name: Optional[str] = None,
    description: Optional[str] = None,
    labels: Optional[list[str]] = None,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = _determine_flavors(studio, local, all, token)

    if all or local:
        try:
            catalog.edit_dataset(name, new_name, description, labels)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)

    if (all or studio) and token:
        edit_studio_dataset(team, name, new_name, description, labels)


def dataset_stats(
    catalog: "Catalog",
    name: str,
    version: int,
    show_bytes=False,
    si=False,
):
    stats = catalog.dataset_stats(name, version)

    if stats:
        print(f"Number of objects: {stats.num_objects}")
        if show_bytes:
            print(f"Total objects size: {stats.size}")
        else:
            print(f"Total objects size: {utils.sizeof_fmt(stats.size, si=si): >7}")


def du(catalog: "Catalog", sources, show_bytes=False, si=False, **kwargs):
    for path, size in catalog.du(sources, **kwargs):
        if show_bytes:
            print(f"{size} {path}")
        else:
            print(f"{utils.sizeof_fmt(size, si=si): >7} {path}")


def index(
    catalog: "Catalog",
    sources,
    **kwargs,
):
    catalog.index(sources, **kwargs)


def show(
    catalog: "Catalog",
    name: str,
    version: Optional[int] = None,
    limit: int = 10,
    offset: int = 0,
    columns: Sequence[str] = (),
    no_collapse: bool = False,
    schema: bool = False,
) -> None:
    from datachain.lib.dc import DataChain
    from datachain.query.dataset import DatasetQuery
    from datachain.utils import show_records

    dataset = catalog.get_dataset(name)
    dataset_version = dataset.get_version(version or dataset.latest_version)

    query = (
        DatasetQuery(name=name, version=version, catalog=catalog)
        .select(*columns)
        .limit(limit)
        .offset(offset)
    )
    records = query.to_db_records()
    show_records(records, collapse_columns=not no_collapse)
    if schema and dataset_version.feature_schema:
        print("\nSchema:")
        session = Session.get(catalog=catalog)
        dc = DataChain.from_dataset(name=name, version=version, session=session)
        dc.print_schema()


def query(
    catalog: "Catalog",
    script: str,
    parallel: Optional[int] = None,
    params: Optional[dict[str, str]] = None,
) -> None:
    from datachain.data_storage import JobQueryType, JobStatus

    with open(script, encoding="utf-8") as f:
        script_content = f.read()

    if parallel is not None:
        # This also sets this environment variable for any subprocesses
        os.environ["DATACHAIN_SETTINGS_PARALLEL"] = str(parallel)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_executable = sys.executable

    job_id = catalog.metastore.create_job(
        name=os.path.basename(script),
        query=script_content,
        query_type=JobQueryType.PYTHON,
        python_version=python_version,
        params=params,
    )

    try:
        catalog.query(
            script_content,
            python_executable=python_executable,
            params=params,
            job_id=job_id,
        )
    except Exception as e:
        error_message = str(e)
        error_stack = traceback.format_exc()
        catalog.metastore.set_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_message,
            error_stack=error_stack,
        )
        raise
    catalog.metastore.set_job_status(job_id, JobStatus.COMPLETE)


def clear_cache(catalog: "Catalog"):
    catalog.cache.clear()


def garbage_collect(catalog: "Catalog"):
    temp_tables = catalog.get_temp_table_names()
    if not temp_tables:
        print("Nothing to clean up.")
    else:
        print(f"Garbage collecting {len(temp_tables)} tables.")
        catalog.cleanup_tables(temp_tables)


def completion(shell: str) -> str:
    return shtab.complete(
        get_parser(),
        shell=shell,
    )


def _determine_flavors(studio: bool, local: bool, all: bool, token: Optional[str]):
    if studio and not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain studio login'."
        )

    if local or studio:
        all = False

    all = all and not (local or studio)

    return all, local, studio


def main(argv: Optional[list[str]] = None) -> int:  # noqa: C901, PLR0912, PLR0915
    # Required for Windows multiprocessing support
    freeze_support()

    parser = get_parser()
    args = parser.parse_args(argv)

    if args.command == "internal-run-udf":
        from datachain.query.dispatch import udf_entrypoint

        return udf_entrypoint()

    if args.command == "internal-run-udf-worker":
        from datachain.query.dispatch import udf_worker_entrypoint

        return udf_worker_entrypoint()

    from .catalog import get_catalog

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
        if args.command == "cp":
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
        elif args.command == "clone":
            catalog.clone(
                args.sources,
                args.output,
                force=bool(args.force),
                update=bool(args.update),
                recursive=bool(args.recursive),
                no_glob=args.no_glob,
                no_cp=args.no_cp,
                edatachain=args.edatachain,
                edatachain_file=args.edatachain_file,
            )
        elif args.command in ("datasets", "ds"):
            if args.datasets_cmd == "pull":
                catalog.pull_dataset(
                    args.dataset,
                    args.output,
                    local_ds_name=args.local_name,
                    local_ds_version=args.local_version,
                    cp=args.cp,
                    force=bool(args.force),
                    edatachain=args.edatachain,
                    edatachain_file=args.edatachain_file,
                )
            elif args.datasets_cmd == "edit":
                edit_dataset(
                    catalog,
                    args.name,
                    new_name=args.new_name,
                    description=args.description,
                    labels=args.labels,
                    studio=args.studio,
                    local=args.local,
                    all=args.all,
                    team=args.team,
                )
            elif args.datasets_cmd == "ls":
                datasets(
                    catalog=catalog,
                    studio=args.studio,
                    local=args.local,
                    all=args.all,
                    team=args.team,
                )
            elif args.datasets_cmd in ("rm", "remove"):
                rm_dataset(
                    catalog,
                    args.name,
                    version=args.version,
                    force=args.force,
                    studio=args.studio,
                    local=args.local,
                    all=args.all,
                    team=args.team,
                )
            elif args.datasets_cmd == "stats":
                dataset_stats(
                    catalog,
                    args.name,
                    args.version,
                    show_bytes=args.bytes,
                    si=args.si,
                )
            else:
                raise Exception(f"Unexpected command {args.datasets_cmd}")
        elif args.command == "ls":
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

        elif args.command == "show":
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

        elif args.command == "du":
            du(
                catalog,
                args.sources,
                show_bytes=args.bytes,
                depth=args.depth,
                si=args.si,
                update=bool(args.update),
                client_config=client_config,
            )
        elif args.command == "find":
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
        elif args.command == "index":
            index(
                catalog,
                args.sources,
                update=bool(args.update),
            )
        elif args.command == "completion":
            print(completion(args.shell))
        elif args.command == "query":
            query(
                catalog,
                args.script,
                parallel=args.parallel,
                params=args.param,
            )
        elif args.command == "clear-cache":
            clear_cache(catalog)
        elif args.command == "gc":
            garbage_collect(catalog)
        elif args.command == "studio":
            process_studio_cli_args(args)
        else:
            print(f"invalid command: {args.command}", file=sys.stderr)
            return 1
        return 0
    except BrokenPipeError as exc:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        # See: https://docs.python.org/3/library/signal.html#note-on-sigpipe
        error = str(exc)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141  # 128 + 13 (SIGPIPE)
    except (KeyboardInterrupt, Exception) as exc:
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
        return 1
    finally:
        telemetry.send_cli_call(args.command, error=error)
