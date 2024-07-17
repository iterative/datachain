from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, JSON, Boolean, DateTime, Float, Integer, String

SQL_TO_PYTHON = {
    String: str,
    Integer: int,
    Float: float,
    Boolean: bool,
    DateTime: datetime,
    ARRAY: list,
    JSON: dict,
}


def sql_to_python(args_map: dict[str, Any]):
    r = {}
    for k, v in args_map.items():
        x = type(v.type)
        type_ = SQL_TO_PYTHON.get(x, str)
        r[k] = type_
    return r
