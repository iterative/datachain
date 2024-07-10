import json
import os
from typing import Optional

params_cache: Optional[dict[str, str]] = None


def param(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get query parameter."""
    if not isinstance(key, str):
        raise TypeError("Param key must be a string")

    global params_cache  # noqa: PLW0603
    if params_cache is None:
        if env_params := os.getenv("DATACHAIN_QUERY_PARAMS"):
            try:
                params_parsed = json.loads(env_params)
            except (TypeError, ValueError):
                raise ValueError("Invalid params provided") from None
            if isinstance(params_parsed, dict):
                params_cache = params_parsed
            else:
                raise ValueError("Invalid params provided")
        else:
            params_cache = {}

    return params_cache.get(key, default)
