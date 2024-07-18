from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, JSON, Boolean, DateTime, Float, Integer, String

from datachain.data_storage.sqlite import Column

SQL_TO_PYTHON = {
    String: str,
    Integer: int,
    Float: float,
    Boolean: bool,
    DateTime: datetime,
    ARRAY: list,
    JSON: dict,
}


def sql_to_python(args_map: dict[str, Column]) -> dict[str, Any]:
    return {
        k: SQL_TO_PYTHON.get(type(v.type), str)  # type: ignore[union-attr]
        for k, v in args_map.items()
    }
