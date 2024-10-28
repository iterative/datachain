import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from sqlalchemy import BindParameter, ColumnElement, desc

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query.schema import Column, ColumnMeta

from .inner.base import Function

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema


ColT = Union[str, ColumnElement, "Func"]


@dataclass
class Window:
    """Represents a window specification for SQL window functions."""

    partition_by: str
    order_by: str
    desc: bool = False


def window(partition_by: str, order_by: str, desc: bool = False) -> Window:
    """
    Defines a window specification for SQL window functions.

    The `window` function specifies how to partition and order the result set
    for the associated window function. It is used to define the scope of the rows
    that the window function will operate on.

    Args:
        partition_by (str): The column name by which to partition the result set.
                            Rows with the same value in the partition column
                            will be grouped together for the window function.
        order_by (str): The column name by which to order the rows
                        within each partition. This determines the sequence in which
                        the window function is applied.
        desc (bool, optional): If True, the rows will be ordered in descending order.
                               Defaults to False, which orders the rows
                               in ascending order.

    Returns:
        Window: A Window object representing the window specification.

    Example:
        ```py
        window = func.window(partition_by="signal.category", order_by="created_at")
        dc.mutate(
            row_number=func.row_number().over(window),
        )
        ```
    """
    return Window(
        ColumnMeta.to_db_name(partition_by),
        ColumnMeta.to_db_name(order_by),
        desc,
    )


class Func(Function):
    """Represents a function to be applied to a column in a SQL query."""

    def __init__(
        self,
        name: str,
        inner: Callable,
        cols: Optional[Sequence[ColT]] = None,
        args: Optional[Sequence[Any]] = None,
        result_type: Optional["DataType"] = None,
        is_array: bool = False,
        is_window: bool = False,
        window: Optional[Window] = None,
        label: Optional[str] = None,
    ) -> None:
        self.name = name
        self.inner = inner
        self.cols = cols or []
        self.args = args or []
        self.result_type = result_type
        self.is_array = is_array
        self.is_window = is_window
        self.window = window
        self.col_label = label

    def __str__(self) -> str:
        return self.name + "()"

    def over(self, window: Window) -> "Func":
        if not self.is_window:
            raise DataChainParamsError(f"{self} doesn't support window (over())")

        return Func(
            "over",
            self.inner,
            self.cols,
            self.args,
            self.result_type,
            self.is_array,
            self.is_window,
            window,
            self.col_label,
        )

    @property
    def _db_cols(self) -> Sequence[ColT]:
        return (
            [
                col
                if isinstance(col, (Func, BindParameter))
                else ColumnMeta.to_db_name(
                    col.name if isinstance(col, ColumnElement) else col
                )
                for col in self.cols
            ]
            if self.cols
            else []
        )

    def _db_col_type(self, signals_schema: "SignalSchema") -> Optional["DataType"]:
        if not self._db_cols:
            return None

        col_type: type = get_db_col_type(signals_schema, self._db_cols[0])
        for col in self._db_cols[1:]:
            if get_db_col_type(signals_schema, col) != col_type:
                raise DataChainColumnError(
                    str(self),
                    "Columns must have the same type to infer result type",
                )

        return list[col_type] if self.is_array else col_type  # type: ignore[valid-type]

    def __add__(self, other: Union[ColT, float]) -> "Func":
        return sum(self, other)

    def __radd__(self, other: Union[ColT, float]) -> "Func":
        return sum(other, self)

    def __sub__(self, other: Union[ColT, float]) -> "Func":
        return sub(self, other)

    def __rsub__(self, other: Union[ColT, float]) -> "Func":
        return sub(other, self)

    def __mul__(self, other: Union[ColT, float]) -> "Func":
        return multiply(self, other)

    def __rmul__(self, other: Union[ColT, float]) -> "Func":
        return multiply(other, self)

    def __truediv__(self, other: Union[ColT, float]) -> "Func":
        return divide(self, other)

    def __rtruediv__(self, other: Union[ColT, float]) -> "Func":
        return divide(other, self)

    def __gt__(self, other: Union[ColT, float]) -> "Func":
        return gt(self, other)

    def __lt__(self, other: Union[ColT, float]) -> "Func":
        return lt(self, other)

    def label(self, label: str) -> "Func":
        return Func(
            self.name,
            self.inner,
            self.cols,
            self.args,
            self.result_type,
            self.is_array,
            self.is_window,
            self.window,
            label,
        )

    def get_col_name(self, label: Optional[str] = None) -> str:
        if label:
            return label
        if self.col_label:
            return self.col_label
        if (db_cols := self._db_cols) and len(db_cols) == 1:
            if isinstance(db_cols[0], str):
                return db_cols[0]
            if isinstance(db_cols[0], Column):
                return db_cols[0].name
            if isinstance(db_cols[0], Func):
                return db_cols[0].get_col_name()
        return self.name

    def get_result_type(
        self, signals_schema: Optional["SignalSchema"] = None
    ) -> "DataType":
        if self.result_type:
            return self.result_type

        if signals_schema and (col_type := self._db_col_type(signals_schema)):
            return col_type

        raise DataChainColumnError(
            str(self),
            "Column name is required to infer result type",
        )

    def get_column(
        self,
        signals_schema: Optional["SignalSchema"] = None,
        label: Optional[str] = None,
        table: Optional["TableClause"] = None,
    ) -> Column:
        col_type = self.get_result_type(signals_schema)
        sql_type = python_to_sql(col_type)

        def get_col(col: ColT) -> ColT:
            if isinstance(col, Func):
                return col.get_column(signals_schema, table=table)
            if isinstance(col, str):
                column = Column(col, sql_type)
                column.table = table
                return column
            return col

        cols = [get_col(col) for col in self._db_cols]
        func_col = self.inner(*cols, *self.args)

        if self.is_window:
            if not self.window:
                raise DataChainParamsError(
                    f"Window function {self} requires over() clause with a window spec",
                )
            func_col = func_col.over(
                partition_by=self.window.partition_by,
                order_by=(
                    desc(self.window.order_by)
                    if self.window.desc
                    else self.window.order_by
                ),
            )

        func_col.type = sql_type() if inspect.isclass(sql_type) else sql_type

        if col_name := self.get_col_name(label):
            func_col = func_col.label(col_name)

        return func_col


def get_db_col_type(signals_schema: "SignalSchema", col: ColT) -> "DataType":
    if isinstance(col, Func):
        return col.get_result_type(signals_schema)

    return signals_schema.get_column_type(
        col.name if isinstance(col, ColumnElement) else col
    )


def sum(*args: Union[ColT, float]) -> Func:
    """Computes the sum of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func("sum", lambda a1, a2: a1 + a2, cols=cols, args=func_args)


def sub(*args: Union[ColT, float]) -> Func:
    """Computes the diff of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func("sub", lambda a1, a2: a1 - a2, cols=cols, args=func_args)


def multiply(*args: Union[ColT, float]) -> Func:
    """Computes the product of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func("multiply", lambda a1, a2: a1 * a2, cols=cols, args=func_args)


def divide(*args: Union[ColT, float]) -> Func:
    """Computes the division of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func(
        "divide", lambda a1, a2: a1 / a2, cols=cols, args=func_args, result_type=float
    )


def gt(*args: Union[ColT, float]) -> Func:
    """Computes the greater than comparison of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func(
        "gt", lambda a1, a2: a1 > a2, cols=cols, args=func_args, result_type=bool
    )


def lt(*args: Union[ColT, float]) -> Func:
    """Computes the less than comparison of the column."""
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, (int, float)):
            func_args.append(arg)
        else:
            cols.append(arg)

    return Func(
        "lt", lambda a1, a2: a1 < a2, cols=cols, args=func_args, result_type=bool
    )
