from typing import Any

from datachain.query.dataset import Column


def sql_to_python(args_map: dict[str, Column]) -> dict[str, Any]:
    res = {}
    for name, sql_exp in args_map.items():
        try:
            type_ = sql_exp.type.python_type
        except NotImplementedError:
            type_ = str
        res[name] = type_

    return res
