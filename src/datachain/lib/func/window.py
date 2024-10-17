from sqlalchemy import func as sa_func

from .func import Func


def row_number() -> Func:
    return Func(inner=sa_func.row_number, result_type=int, is_window=True)


def rank() -> Func:
    return Func(inner=sa_func.rank, result_type=int, is_window=True)


def dense_rank() -> Func:
    return Func(inner=sa_func.dense_rank, result_type=int, is_window=True)


def first(col: str) -> Func:
    return Func(inner=sa_func.first_value, col=col, is_window=True)
