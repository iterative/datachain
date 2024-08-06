from decimal import Decimal
from typing import Any

from sqlalchemy import ColumnElement


def sql_to_python(args_map: dict[str, ColumnElement]) -> dict[str, Any]:
    res = {}
    for name, sql_exp in args_map.items():
        try:
            type_ = sql_exp.type.python_type
            if type_ == Decimal:
                type_ = float
        except NotImplementedError:
            type_ = str
        res[name] = type_

    return res
