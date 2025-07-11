from decimal import Decimal
from typing import Any

from sqlalchemy import ColumnElement


def sql_to_python(sql_exp: ColumnElement) -> Any:
    try:
        type_ = sql_exp.type.python_type
        if type_ == Decimal:
            type_ = float
        elif type_ is list:
            if hasattr(sql_exp.type, "item_type") and hasattr(
                sql_exp.type.item_type, "python_type"
            ):
                type_ = list[sql_exp.type.item_type.python_type]
            else:
                type_ = list
    except NotImplementedError:
        type_ = str
    return type_
