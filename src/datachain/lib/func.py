from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy import func

if TYPE_CHECKING:
    from datachain import DataType


class Func:
    result_type: Optional["DataType"] = None

    def __init__(
        self,
        inner: Callable,
        cols: tuple[str, ...],
        result_type: Optional["DataType"] = None,
    ) -> None:
        self.inner = inner
        self.cols = [col.replace(".", "__") for col in cols]
        self.result_type = result_type


def sum(*cols: str) -> Func:
    return Func(inner=func.sum, cols=cols)


def count(*cols: str) -> Func:
    return Func(inner=func.count, cols=cols, result_type=int)
