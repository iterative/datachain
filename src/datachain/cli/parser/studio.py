def add_studio_parser(subparsers, parent_parser) -> None:
    studio_help = "Manage Studio authentication"
    studio_description = (
        "Manage authentication and settings for Studio. "
        "Configure tokens for sharing datasets and using Studio features."
    )

    studio_parser = subparsers.add_parser(
        "studio",
        parents=[parent_parser],
        description=studio_description,
        help=studio_help,
    )
    studio_subparser = studio_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain studio CMD --help` to display command-specific help",
    )

    studio_login_help = "Authenticate with Studio"
    studio_login_description = (
        "Authenticate with Studio using default scopes. "
        "A random name will be assigned as the token name if not specified."
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
        help="Hostname of the Studio instance",
    )
    login_parser.add_argument(
        "-s",
        "--scopes",
        action="store",
        default=None,
        help="Authentication token scopes",
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

    studio_logout_help = "Log out from Studio"
    studio_logout_description = (
        "Remove the Studio authentication token from global config."
    )

    studio_subparser.add_parser(
        "logout",
        parents=[parent_parser],
        description=studio_logout_description,
        help=studio_logout_help,
    )

    studio_team_help = "Set default team for Studio operations"
    studio_team_description = "Set the default team for Studio operations."

    team_parser = studio_subparser.add_parser(
        "team",
        parents=[parent_parser],
        description=studio_team_description,
        help=studio_team_help,
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

    studio_token_help = "View Studio authentication token"  # noqa: S105
    studio_token_description = "Display the current authentication token for Studio."  # noqa: S105

    studio_subparser.add_parser(
        "token",
        parents=[parent_parser],
        description=studio_token_description,
        help=studio_token_help,
    )

    studio_ls_dataset_help = "List available Studio datasets"
    studio_ls_dataset_description = (
        "List all datasets available in Studio, showing dataset names "
        "and version counts."
    )

    ls_dataset_parser = studio_subparser.add_parser(
        "dataset",
        parents=[parent_parser],
        description=studio_ls_dataset_description,
        help=studio_ls_dataset_help,
    )
    ls_dataset_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to list datasets for (default: from config)",
    )
