import shtab

from datachain.cli.parser.utils import CustomHelpFormatter


def add_auth_parser(subparsers, parent_parser) -> None:
    from dvc_studio_client.auth import AVAILABLE_SCOPES

    auth_help = "Manage Studio authentication"
    auth_description = "Manage authentication and settings for Studio. "

    auth_parser = subparsers.add_parser(
        "auth",
        parents=[parent_parser],
        description=auth_description,
        help=auth_help,
        formatter_class=CustomHelpFormatter,
    )
    auth_subparser = auth_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain auth CMD --help` to display command-specific help",
    )

    auth_login_help = "Authenticate with Studio"
    auth_login_description = (
        "Authenticate with Studio using default scopes. "
        "A random name will be assigned if the token name is not specified."
    )

    allowed_scopes = ", ".join(AVAILABLE_SCOPES)
    login_parser = auth_subparser.add_parser(
        "login",
        parents=[parent_parser],
        description=auth_login_description,
        help=auth_login_help,
        formatter_class=CustomHelpFormatter,
    )

    login_parser.add_argument(
        "-H",
        "--hostname",
        action="store",
        default=None,
        help="Hostname of the Studio instance",
    )
    login_parser.add_argument(
        "-s",
        "--scopes",
        action="store",
        default=None,
        help=f"Authentication token scopes. Allowed scopes: {allowed_scopes}",
    )

    login_parser.add_argument(
        "-n",
        "--name",
        action="store",
        default=None,
        help="Authentication token name (shown in Studio profile)",
    )

    login_parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Use code-based authentication without browser",
    )
    login_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Save the token in the local project config",
    )

    auth_logout_help = "Log out from Studio"
    auth_logout_description = (
        "Remove the Studio authentication token from global config."
    )

    logout_parser = auth_subparser.add_parser(
        "logout",
        parents=[parent_parser],
        description=auth_logout_description,
        help=auth_logout_help,
        formatter_class=CustomHelpFormatter,
    )
    logout_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Remove the token from the local project config",
    )

    auth_team_help = "Set or show default team for Studio operations"
    auth_team_description = (
        "Set or show the default team for Studio operations. "
        "This will be used globally by default. "
        "Use --local to set the team locally for the current project. "
        "If no team name is provided, the default team will be shown."
    )

    team_parser = auth_subparser.add_parser(
        "team",
        parents=[parent_parser],
        description=auth_team_description,
        help=auth_team_help,
        formatter_class=CustomHelpFormatter,
    )
    team_parser.add_argument(
        "team_name",
        action="store",
        default=None,
        nargs="?",
        help="Name of the team to set as default",
    )
    team_parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Set team locally for the current project",
    )

    auth_token_help = "View Studio authentication token"  # noqa: S105
    auth_token_description = "Display the current authentication token for Studio."  # noqa: S105

    auth_subparser.add_parser(
        "token",
        parents=[parent_parser],
        description=auth_token_description,
        help=auth_token_help,
        formatter_class=CustomHelpFormatter,
    )


def add_storage_parser(subparsers, parent_parser) -> None:
    storage_cp_help = "Copy storage contents"
    storage_cp_description = (
        "Copy storage files and directories between cloud and local storage"
    )

    storage_cp_parser = subparsers.add_parser(
        "cp",
        parents=[parent_parser],
        description=storage_cp_description,
        help=storage_cp_help,
        formatter_class=CustomHelpFormatter,
    )

    storage_cp_parser.add_argument(
        "source_path",
        action="store",
        help="Path to the source file or directory to copy",
    ).complete = shtab.DIR  # type: ignore[attr-defined]

    storage_cp_parser.add_argument(
        "destination_path",
        action="store",
        help="Path to the destination file or directory to copy",
    ).complete = shtab.DIR  # type: ignore[attr-defined]

    storage_cp_parser.add_argument(
        "-r",
        "-R",
        "--recursive",
        action="store_true",
        help="Copy directories recursively",
    )

    storage_cp_parser.add_argument(
        "--team",
        action="store",
        help="Team name to use the credentials from.",
    )

    storage_cp_parser.add_argument(
        "-s",
        "--studio-cloud-auth",
        default=False,
        action="store_true",
        help="Use credentials from Studio for cloud operations (Default: False)",
    )

    storage_cp_parser.add_argument(
        "--update",
        action="store_true",
        help="Update cached list of files for the source in datachain cache",
    )

    storage_cp_parser.add_argument(
        "--anon",
        action="store_true",
        help="Use anonymous access for cloud operations (Default: False)",
    )

    mv_parser = subparsers.add_parser(
        "mv",
        parents=[parent_parser],
        description="Move storage files and directories through Studio",
        help="Move storage files and directories through Studio",
        formatter_class=CustomHelpFormatter,
    )
    mv_parser.add_argument(
        "path",
        action="store",
        help="Path to the storage file or directory to move",
    )
    mv_parser.add_argument(
        "new_path",
        action="store",
        help="New path to the storage file or directory to move",
    )
    mv_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Move recursively",
    )
    mv_parser.add_argument(
        "--team",
        action="store",
        help="Team name to use the credentials from.",
    )

    mv_parser.add_argument(
        "-s",
        "--studio-cloud-auth",
        default=False,
        action="store_true",
        help="Use credentials from Studio for cloud operations (Default: False)",
    )

    rm_parser = subparsers.add_parser(
        "rm",
        parents=[parent_parser],
        description="Delete storage files and directories through Studio",
        help="Delete storage files and directories through Studio",
        formatter_class=CustomHelpFormatter,
    )
    rm_parser.add_argument(
        "path",
        action="store",
        help="Path to the storage file or directory to delete",
    )
    rm_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Delete recursively",
    )
    rm_parser.add_argument(
        "--team",
        action="store",
        help="Team name to use the credentials from.",
    )
    rm_parser.add_argument(
        "-s",
        "--studio-cloud-auth",
        default=False,
        action="store_true",
        help="Use credentials from Studio for cloud operations (Default: False)",
    )
