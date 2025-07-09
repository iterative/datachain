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
    storage_help = "Manage storage"
    storage_description = "Manage storage through Studio"

    storage_parser = subparsers.add_parser(
        "storage",
        parents=[parent_parser],
        description=storage_description,
        help=storage_help,
        formatter_class=CustomHelpFormatter,
    )

    storage_subparser = storage_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain storage CMD --help` to display command-specific help",
    )

    storage_delete_help = "Delete storage contents"
    storage_delete_description = "Delete storage files and directories through Studio"

    storage_delete_parser = storage_subparser.add_parser(
        "rm",
        parents=[parent_parser],
        description=storage_delete_description,
        help=storage_delete_help,
        formatter_class=CustomHelpFormatter,
    )

    storage_delete_parser.add_argument(
        "path",
        action="store",
        help="Path to the storage file or directory to delete",
    )

    storage_delete_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Delete recursively",
    )

    storage_delete_parser.add_argument(
        "--team",
        action="store",
        help="Team name to delete storage contents from",
    )

    storage_move_help = "Move storage contents"
    storage_move_description = "Move storage files and directories through Studio"

    storage_move_parser = storage_subparser.add_parser(
        "mv",
        parents=[parent_parser],
        description=storage_move_description,
        help=storage_move_help,
        formatter_class=CustomHelpFormatter,
    )

    storage_move_parser.add_argument(
        "path",
        action="store",
        help="Path to the storage file or directory to move",
    )

    storage_move_parser.add_argument(
        "new_path",
        action="store",
        help="New path to the storage file or directory to move",
    )

    storage_move_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Move recursively",
    )

    storage_move_parser.add_argument(
        "--team",
        action="store",
        help="Team name to move storage contents from",
    )

    storage_cp_help = "Copy storage contents"
    storage_cp_description = "Copy storage files and directories through Studio"

    storage_cp_parser = storage_subparser.add_parser(
        "cp",
        parents=[parent_parser],
        description=storage_cp_description,
        help=storage_cp_help,
        formatter_class=CustomHelpFormatter,
    )

    storage_cp_parser.add_argument(
        "source_path",
        action="store",
        help="Path to the source file or directory to upload",
    )

    storage_cp_parser.add_argument(
        "destination_path",
        action="store",
        help="Path to the destination file or directory to upload",
    )

    storage_cp_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Upload recursively",
    )

    storage_cp_parser.add_argument(
        "--team",
        action="store",
        help="Team name to upload storage contents to",
    )
