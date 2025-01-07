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
        "dataset",
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
