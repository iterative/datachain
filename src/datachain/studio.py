import os
from typing import TYPE_CHECKING

from datachain.config import Config
from datachain.error import DataChainError
from datachain.utils import STUDIO_URL

if TYPE_CHECKING:
    from argparse import Namespace

POST_LOGIN_MESSAGE = (
    "Once you've logged in, return here "
    "and you'll be ready to start using Datachain with Studio."
)


def process_studio_cli_args(args: "Namespace"):
    if args.cmd == "login":
        return login(args)
    if args.cmd == "logout":
        return logout()
    if args.cmd == "token":
        return token()
    raise DataChainError(f"Unknown command '{args.cmd}'.")


def login(args: "Namespace"):
    from dvc_studio_client.auth import StudioAuthError, get_access_token

    config = Config().read().get("studio", {})
    name = args.name
    hostname = (
        args.hostname
        or os.environ.get("DVC_STUDIO_HOSTNAME")
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
            client_name="Datachain",
            post_login_message=POST_LOGIN_MESSAGE,
        )
    except StudioAuthError as exc:
        raise DataChainError(f"Failed to authenticate with Studio: {exc}") from exc

    config_path = save_config(hostname, access_token)
    print(f"Authentication complete. Saved token to {config_path}.")
    return 0


def logout():
    with Config("global").edit() as conf:
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


def save_config(hostname, token):
    config = Config("global")
    with config.edit() as conf:
        studio_conf = conf.get("studio", {})
        studio_conf["url"] = hostname
        studio_conf["token"] = token
        conf["studio"] = studio_conf

    return config.config_file()
