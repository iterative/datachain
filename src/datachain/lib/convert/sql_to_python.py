from decimal import Decimal
from typing import Any

from sqlalchemy import ColumnElement


def sql_to_python(sql_exp: ColumnElement) -> Any:
    try:
        type_ = sql_exp.type.python_type
        if type_ == Decimal:
            type_ = float
    except NotImplementedError:
        type_ = str
    return type_
