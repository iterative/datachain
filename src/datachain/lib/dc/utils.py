from collections.abc import Sequence
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    TypeVar,
    Union,
)

import sqlalchemy
from sqlalchemy.sql.functions import GenericFunction

from datachain.func.base import Function
from datachain.lib.data_model import DataModel, DataType
from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import DEFAULT_DELIMITER

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")

D = TypeVar("D", bound="DataChain")


def resolve_columns(
    method: "Callable[Concatenate[D, P], D]",
) -> "Callable[Concatenate[D, P], D]":
    """Decorator that resolvs input column names to their actual DB names. This is
    specially important for nested columns as user works with them by using dot
    notation e.g (file.name) but are actually defined with default delimiter
    in DB, e.g file__name.
    If there are any sql functions in arguments, they will just be transferred as is
    to a method.
    """

    @wraps(method)
    def _inner(self: D, *args: "P.args", **kwargs: "P.kwargs") -> D:
        resolved_args = self.signals_schema.resolve(
            *[arg for arg in args if not isinstance(arg, GenericFunction)]  # type: ignore[arg-type]
        ).db_signals()

        for idx, arg in enumerate(args):
            if isinstance(arg, GenericFunction):
                resolved_args.insert(idx, arg)  # type: ignore[arg-type]

        return method(self, *resolved_args, **kwargs)

    return _inner


class DatasetPrepareError(DataChainParamsError):
    def __init__(self, name, msg, output=None):
        name = f" '{name}'" if name else ""
        output = f" output '{output}'" if output else ""
        super().__init__(f"Dataset{name}{output} processing prepare error: {msg}")


class DatasetFromValuesError(DataChainParamsError):
    def __init__(self, name, msg):
        name = f" '{name}'" if name else ""
        super().__init__(f"Dataset{name} from values error: {msg}")


MergeColType = Union[str, Function, sqlalchemy.ColumnElement]


def _validate_merge_on(
    on: Union[MergeColType, Sequence[MergeColType]],
    ds: "DataChain",
) -> Sequence[MergeColType]:
    if isinstance(on, (str, sqlalchemy.ColumnElement)):
        return [on]
    if isinstance(on, Function):
        return [on.get_column(table=ds._query.table)]
    if isinstance(on, Sequence):
        return [
            c.get_column(table=ds._query.table) if isinstance(c, Function) else c
            for c in on
        ]


def _get_merge_error_str(col: MergeColType) -> str:
    if isinstance(col, str):
        return col
    if isinstance(col, Function):
        return f"{col.name}()"
    if isinstance(col, sqlalchemy.Column):
        return col.name.replace(DEFAULT_DELIMITER, ".")
    if isinstance(col, sqlalchemy.ColumnElement) and hasattr(col, "name"):
        return f"{col.name} expression"
    return str(col)


class DatasetMergeError(DataChainParamsError):
    def __init__(
        self,
        on: Union[MergeColType, Sequence[MergeColType]],
        right_on: Optional[Union[MergeColType, Sequence[MergeColType]]],
        msg: str,
    ):
        def _get_str(
            on: Union[MergeColType, Sequence[MergeColType]],
        ) -> str:
            if not isinstance(on, Sequence):
                return str(on)  # type: ignore[unreachable]
            return ", ".join([_get_merge_error_str(col) for col in on])

        on_str = _get_str(on)
        right_on_str = (
            ", right_on='" + _get_str(right_on) + "'"
            if right_on and isinstance(right_on, Sequence)
            else ""
        )
        super().__init__(f"Merge error on='{on_str}'{right_on_str}: {msg}")


OutputType = Union[None, DataType, Sequence[str], dict[str, DataType]]


class Sys(DataModel):
    """Model for internal DataChain signals `id` and `rand`."""

    id: int
    rand: int
