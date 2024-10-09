from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy import func as sa_func

from datachain.query.schema import DEFAULT_DELIMITER
from datachain.sql import functions as dc_func

if TYPE_CHECKING:
    from datachain import DataType


class Func:
    result_type: Optional["DataType"] = None

    def __init__(
        self,
        inner: Callable,
        col: Optional[str] = None,
        result_type: Optional["DataType"] = None,
    ) -> None:
        self.inner = inner
        self.col = col.replace(".", DEFAULT_DELIMITER) if col else None
        self.result_type = result_type


def count(col: Optional[str] = None) -> Func:
    return Func(inner=sa_func.count, col=col, result_type=int)


def sum(col: str) -> Func:
    return Func(inner=sa_func.sum, col=col)


def avg(col: str) -> Func:
    return Func(inner=dc_func.avg, col=col)


def min(col: str) -> Func:
    return Func(inner=sa_func.min, col=col)


def max(col: str) -> Func:
    return Func(inner=sa_func.max, col=col)


def concat(col: str, separator="") -> Func:
    def inner(arg):
        return dc_func.array.group_concat(arg, separator)

    return Func(inner=inner, col=col, result_type=str)
