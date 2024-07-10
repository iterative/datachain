import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from tomlkit import load

if TYPE_CHECKING:
    from tomlkit import TOMLDocument


def read_config(datachain_root: str) -> Optional["TOMLDocument"]:
    config_path = os.path.join(datachain_root, "config")
    try:
        with open(config_path, encoding="utf-8") as f:
            return load(f)
    except FileNotFoundError:
        return None


def get_remote_config(
    config: Optional["TOMLDocument"], remote: str = ""
) -> Mapping[str, str]:
    if config is None:
        return {"type": "local"}
    if not remote:
        try:
            remote = config["core"]["default-remote"]  # type: ignore[index,assignment]
        except KeyError:
            return {"type": "local"}
    try:
        remote_conf: Mapping[str, str] = config["remote"][remote]  # type: ignore[assignment,index]
    except KeyError:
        raise Exception(
            f"missing config section for default remote: remote.{remote}"
        ) from None
    except Exception as exc:
        raise Exception("invalid config") from exc

    if not isinstance(remote_conf, Mapping):
        raise TypeError(f"config section remote.{remote} must be a mapping")

    remote_type = remote_conf.get("type")
    if remote_type not in ("local", "http"):
        raise Exception(
            f'config section remote.{remote} must have "type" with one of: '
            '"local", "http"'
        )

    if remote_type == "http":
        for key in ["url", "username", "token"]:
            try:
                remote_conf[key]
            except KeyError:
                raise Exception(
                    f"config section remote.{remote} of type {remote_type} "
                    f"must contain key {key}"
                ) from None
    elif remote_type != "local":
        raise Exception(
            f"config section remote.{remote} has invalid remote type {remote_type}"
        )
    return remote_conf
