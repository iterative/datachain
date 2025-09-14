import inspect
import sys
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union, get_args, get_origin

if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None

from pydantic import BaseModel
from typing_extensions import Literal as LiteralEx

from datachain.lib.model_store import ModelStore
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int64,
    SQLType,
    String,
)

PYTHON_TO_SQL = {
    int: Int64,
    str: String,
    Literal: String,
    LiteralEx: String,
    Enum: String,
    float: Float,
    bool: Boolean,
    datetime: DateTime,  # Note, list of datetime is not supported yet
    bytes: Binary,  # Note, list of bytes is not supported yet
    list: Array,
    dict: JSON,
}


def _is_union(orig) -> bool:
    if orig == Union:
        return True
    # some code is unreachab in python<3.10
    return UnionType is not None and orig is UnionType  # type: ignore[unreachable]


def python_to_sql(typ):  # noqa: PLR0911
    if inspect.isclass(typ):
        if issubclass(typ, SQLType):
            return typ
        if issubclass(typ, Enum):
            return str

    res = PYTHON_TO_SQL.get(typ)
    if res:
        return res

    orig = get_origin(typ)

    if orig in (Literal, LiteralEx):
        return String

    args = get_args(typ)
    if inspect.isclass(orig) and (issubclass(list, orig) or issubclass(tuple, orig)):
        if args is None:
            raise TypeError(f"Cannot resolve type '{typ}' for flattening features")

        args0 = args[0]
        if ModelStore.is_pydantic(args0):
            return Array(JSON())

        list_type = list_of_args_to_type(args)
        return Array(list_type)

    if orig is Annotated:
        # Ignoring annotations
        return python_to_sql(args[0])

    if inspect.isclass(orig) and issubclass(dict, orig):
        return JSON

    if _is_union(orig):
        if len(args) == 2 and (type(None) in args):
            non_none_arg = args[0] if args[0] is not type(None) else args[1]
            return python_to_sql(non_none_arg)

        if _is_union_str_literal(orig, args):
            return String

        if _is_json_inside_union(orig, args):
            return JSON

    raise TypeError(f"Cannot recognize type {typ}")


def list_of_args_to_type(args) -> SQLType:
    first_type = python_to_sql(args[0])
    for next_arg in args[1:]:
        try:
            next_type = python_to_sql(next_arg)
            if next_type != first_type:
                return JSON()
        except TypeError:
            return JSON()
    return first_type


def _is_json_inside_union(orig, args) -> bool:
    if _is_union(orig) and len(args) >= 2:
        # List in JSON: Union[dict, list[dict]]
        args_no_nones = [arg for arg in args if arg != type(None)]  # noqa: E721
        if len(args_no_nones) == 2:
            args_no_dicts = [arg for arg in args_no_nones if arg is not dict]
            if len(args_no_dicts) == 1 and get_origin(args_no_dicts[0]) is list:
                arg = get_args(args_no_dicts[0])
                if len(arg) == 1 and arg[0] is dict:
                    return True

        # List of objects: Union[MyClass, OtherClass]
        if any(inspect.isclass(arg) and issubclass(arg, BaseModel) for arg in args):
            return True
    return False


def _is_union_str_literal(orig, args) -> bool:
    if not _is_union(orig):
        return False
    return all(arg is str or get_origin(arg) in (Literal, LiteralEx) for arg in args)
