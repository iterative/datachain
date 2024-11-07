import os
from typing import TYPE_CHECKING, Optional

from tabulate import tabulate

from datachain.catalog.catalog import raise_remote_error
from datachain.config import Config, ConfigLevel
from datachain.dataset import QUERY_DATASET_PREFIX
from datachain.error import DataChainError
from datachain.remote.studio import StudioClient
from datachain.utils import STUDIO_URL

if TYPE_CHECKING:
    from argparse import Namespace

POST_LOGIN_MESSAGE = (
    "Once you've logged in, return here "
    "and you'll be ready to start using DataChain with Studio."
)


def process_studio_cli_args(args: "Namespace"):
    if args.cmd == "login":
        return login(args)
    if args.cmd == "logout":
        return logout()
    if args.cmd == "token":
        return token()
    if args.cmd == "datasets":
        rows = [
            {"Name": name, "Version": version}
            for name, version in list_datasets(args.team)
        ]
        print(tabulate(rows, headers="keys"))
        return 0

    if args.cmd == "team":
        return set_team(args)
    raise DataChainError(f"Unknown command '{args.cmd}'.")


def set_team(args: "Namespace"):
    level = ConfigLevel.GLOBAL if args.__dict__.get("global") else ConfigLevel.LOCAL
    config = Config(level)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["team"] = args.team_name
        conf["studio"] = studio_conf

    print(f"Set default team to '{args.team_name}' in {config.config_file()}")


def login(args: "Namespace"):
    from dvc_studio_client.auth import StudioAuthError, get_access_token

    config = Config().read().get("studio", {})
    name = args.name
    hostname = (
        args.hostname
        or os.environ.get("DVC_STUDIO_URL")
        or config.get("url")
        or STUDIO_URL
    )
    scopes = args.scopes

    if config.get("url", hostname) == hostname and "token" in config:
        raise DataChainError(
            "Token already exists. "
            "To login with a different token, "
            "logout using `datachain studio logout`."
        )

    open_browser = not args.no_open
    try:
        _, access_token = get_access_token(
            token_name=name,
            hostname=hostname,
            scopes=scopes,
            open_browser=open_browser,
            client_name="DataChain",
            post_login_message=POST_LOGIN_MESSAGE,
        )
    except StudioAuthError as exc:
        raise DataChainError(f"Failed to authenticate with Studio: {exc}") from exc

    config_path = save_config(hostname, access_token)
    print(f"Authentication complete. Saved token to {config_path}.")
    return 0


def logout():
    with Config(ConfigLevel.GLOBAL).edit() as conf:
        token = conf.get("studio", {}).get("token")
        if not token:
            raise DataChainError(
                "Not logged in to Studio. Log in with 'datachain studio login'."
            )

        del conf["studio"]["token"]

    print("Logged out from Studio. (you can log back in with 'datachain studio login')")


def token():
    config = Config().read().get("studio", {})
    token = config.get("token")
    if not token:
        raise DataChainError(
            "Not logged in to Studio. Log in with 'datachain studio login'."
        )

    print(token)


def list_datasets(team: Optional[str] = None):
    client = StudioClient(team=team)
    response = client.ls_datasets()
    if not response.ok:
        raise_remote_error(response.message)
    if not response.data:
        return

    for d in response.data:
        name = d.get("name")
        if name and name.startswith(QUERY_DATASET_PREFIX):
            continue

        for v in d.get("versions", []):
            version = v.get("version")
            yield (name, version)


def save_config(hostname, token):
    config = Config(ConfigLevel.GLOBAL)
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["url"] = hostname
        studio_conf["token"] = token
        conf["studio"] = studio_conf

    return config.config_file()
