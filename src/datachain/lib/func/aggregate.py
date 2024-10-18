from typing import Optional

from sqlalchemy import func as sa_func

from datachain.sql import functions as dc_func

from .func import Func


def count(col: Optional[str] = None) -> Func:
    return Func(inner=sa_func.count, col=col, result_type=int)


def sum(col: str) -> Func:
    return Func(inner=sa_func.sum, col=col)


def avg(col: str) -> Func:
    return Func(inner=dc_func.aggregate.avg, col=col, result_type=float)


def min(col: str) -> Func:
    return Func(inner=sa_func.min, col=col)


def max(col: str) -> Func:
    return Func(inner=sa_func.max, col=col)


def any_value(col: str) -> Func:
    return Func(inner=dc_func.aggregate.any_value, col=col)


def collect(col: str) -> Func:
    return Func(inner=dc_func.aggregate.collect, col=col, is_array=True)


def concat(col: str, separator="") -> Func:
    def inner(arg):
        return dc_func.aggregate.group_concat(arg, separator)

    return Func(inner=inner, col=col, result_type=str)


def row_number() -> Func:
    return Func(inner=sa_func.row_number, result_type=int, is_window=True)


def rank() -> Func:
    return Func(inner=sa_func.rank, result_type=int, is_window=True)


def dense_rank() -> Func:
    return Func(inner=sa_func.dense_rank, result_type=int, is_window=True)


def first(col: str) -> Func:
    return Func(inner=sa_func.first_value, col=col, is_window=True)
