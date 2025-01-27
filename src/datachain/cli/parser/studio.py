def add_auth_parser(subparsers, parent_parser) -> None:
    from dvc_studio_client.auth import AVAILABLE_SCOPES

    auth_help = "Manage Studio authentication"
    auth_description = "Manage authentication and settings for Studio. "

    auth_parser = subparsers.add_parser(
        "auth",
        parents=[parent_parser],
        description=auth_description,
        help=auth_help,
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

    auth_logout_help = "Log out from Studio"
    auth_logout_description = (
        "Remove the Studio authentication token from global config."
    )

    auth_subparser.add_parser(
        "logout",
        parents=[parent_parser],
        description=auth_logout_description,
        help=auth_logout_help,
    )

    auth_team_help = "Set default team for Studio operations"
    auth_team_description = "Set the default team for Studio operations."

    team_parser = auth_subparser.add_parser(
        "team",
        parents=[parent_parser],
        description=auth_team_description,
        help=auth_team_help,
    )
    team_parser.add_argument(
        "team_name",
        action="store",
        help="Name of the team to set as default",
    )
    team_parser.add_argument(
        "--global",
        action="store_true",
        default=False,
        help="Set team globally for all projects",
    )

    auth_token_help = "View Studio authentication token"  # noqa: S105
    auth_token_description = "Display the current authentication token for Studio."  # noqa: S105

    auth_subparser.add_parser(
        "token",
        parents=[parent_parser],
        description=auth_token_description,
        help=auth_token_help,
    )
