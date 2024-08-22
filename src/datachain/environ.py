import os
from typing import TypeVar, Union, overload

T = TypeVar("T")


@overload
def get_bool_env(name: str) -> bool: ...


@overload
def get_bool_env(name: str, undefined: "T" = ...) -> Union["T", bool]: ...


def get_bool_env(name, undefined=False):
    value_str = os.getenv(name, None)
    if not value_str:
        return undefined

    try:
        return int(value_str) != 0
    except ValueError:
        return value_str.lower().strip() in ("true", "on", "ok", "y", "yes", "1")


def get_envs_by_prefix(prefix: str) -> dict[str, str]:
    """
    Function that searches env variables by some name prefix and returns
    the ones found, but with prefix being excluded from it's names
    """
    variables: dict[str, str] = {}
    for env_name, env_value in os.environ.items():
        if env_name.startswith(prefix):
            variables[env_name[len(prefix) :]] = env_value

    return variables
