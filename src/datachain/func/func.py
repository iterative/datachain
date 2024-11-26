import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from sqlalchemy import BindParameter, ColumnElement, desc

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query.schema import Column, ColumnMeta

from .base import Function

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema

    from .window import Window


ColT = Union[str, ColumnElement, "Func"]


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
        window: Optional["Window"] = None,
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

    def over(self, window: "Window") -> "Func":
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
        return math_add(self, other)

    def __radd__(self, other: Union[ColT, float]) -> "Func":
        return math_add(other, self)

    def __sub__(self, other: Union[ColT, float]) -> "Func":
        return math_sub(self, other)

    def __rsub__(self, other: Union[ColT, float]) -> "Func":
        return math_sub(other, self)

    def __mul__(self, other: Union[ColT, float]) -> "Func":
        return math_mul(self, other)

    def __rmul__(self, other: Union[ColT, float]) -> "Func":
        return math_mul(other, self)

    def __truediv__(self, other: Union[ColT, float]) -> "Func":
        return math_truediv(self, other)

    def __rtruediv__(self, other: Union[ColT, float]) -> "Func":
        return math_truediv(other, self)

    def __floordiv__(self, other: Union[ColT, float]) -> "Func":
        return math_floordiv(self, other)

    def __rfloordiv__(self, other: Union[ColT, float]) -> "Func":
        return math_floordiv(other, self)

    def __mod__(self, other: Union[ColT, float]) -> "Func":
        return math_mod(self, other)

    def __rmod__(self, other: Union[ColT, float]) -> "Func":
        return math_mod(other, self)

    def __pow__(self, other: Union[ColT, float]) -> "Func":
        return math_pow(self, other)

    def __rpow__(self, other: Union[ColT, float]) -> "Func":
        return math_pow(other, self)

    def __lshift__(self, other: Union[ColT, float]) -> "Func":
        return math_lshift(self, other)

    def __rlshift__(self, other: Union[ColT, float]) -> "Func":
        return math_lshift(other, self)

    def __rshift__(self, other: Union[ColT, float]) -> "Func":
        return math_rshift(self, other)

    def __rrshift__(self, other: Union[ColT, float]) -> "Func":
        return math_rshift(other, self)

    def __and__(self, other: Union[ColT, float]) -> "Func":
        return math_and(self, other)

    def __rand__(self, other: Union[ColT, float]) -> "Func":
        return math_and(other, self)

    def __or__(self, other: Union[ColT, float]) -> "Func":
        return math_or(self, other)

    def __ror__(self, other: Union[ColT, float]) -> "Func":
        return math_or(other, self)

    def __xor__(self, other: Union[ColT, float]) -> "Func":
        return math_xor(self, other)

    def __rxor__(self, other: Union[ColT, float]) -> "Func":
        return math_xor(other, self)

    def __lt__(self, other: Union[ColT, float]) -> "Func":
        return math_lt(self, other)

    def __le__(self, other: Union[ColT, float]) -> "Func":
        return math_le(self, other)

    def __eq__(self, other):
        return math_eq(self, other)

    def __ne__(self, other):
        return math_ne(self, other)

    def __gt__(self, other: Union[ColT, float]) -> "Func":
        return math_gt(self, other)

    def __ge__(self, other: Union[ColT, float]) -> "Func":
        return math_ge(self, other)

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


def math_func(
    name: str,
    inner: Callable,
    params: Sequence[Union[ColT, float]],
    result_type: Optional["DataType"] = None,
) -> Func:
    """Returns math function from the columns."""
    cols, args = [], []
    for arg in params:
        if isinstance(arg, (int, float)):
            args.append(arg)
        else:
            cols.append(arg)
    return Func(name, inner, cols=cols, args=args, result_type=result_type)


def math_add(*args: Union[ColT, float]) -> Func:
    """Computes the sum of the column."""
    return math_func("add", lambda a1, a2: a1 + a2, args)


def math_sub(*args: Union[ColT, float]) -> Func:
    """Computes the diff of the column."""
    return math_func("sub", lambda a1, a2: a1 - a2, args)


def math_mul(*args: Union[ColT, float]) -> Func:
    """Computes the product of the column."""
    return math_func("mul", lambda a1, a2: a1 * a2, args)


def math_truediv(*args: Union[ColT, float]) -> Func:
    """Computes the division of the column."""
    return math_func("div", lambda a1, a2: a1 / a2, args, result_type=float)


def math_floordiv(*args: Union[ColT, float]) -> Func:
    """Computes the floor division of the column."""
    return math_func("floordiv", lambda a1, a2: a1 // a2, args, result_type=float)


def math_mod(*args: Union[ColT, float]) -> Func:
    """Computes the modulo of the column."""
    return math_func("mod", lambda a1, a2: a1 % a2, args, result_type=float)


def math_pow(*args: Union[ColT, float]) -> Func:
    """Computes the power of the column."""
    return math_func("pow", lambda a1, a2: a1**a2, args, result_type=float)


def math_lshift(*args: Union[ColT, float]) -> Func:
    """Computes the left shift of the column."""
    return math_func("lshift", lambda a1, a2: a1 << a2, args, result_type=int)


def math_rshift(*args: Union[ColT, float]) -> Func:
    """Computes the right shift of the column."""
    return math_func("rshift", lambda a1, a2: a1 >> a2, args, result_type=int)


def math_and(*args: Union[ColT, float]) -> Func:
    """Computes the logical AND of the column."""
    return math_func("and", lambda a1, a2: a1 & a2, args, result_type=bool)


def math_or(*args: Union[ColT, float]) -> Func:
    """Computes the logical OR of the column."""
    return math_func("or", lambda a1, a2: a1 | a2, args, result_type=bool)


def math_xor(*args: Union[ColT, float]) -> Func:
    """Computes the logical XOR of the column."""
    return math_func("xor", lambda a1, a2: a1 ^ a2, args, result_type=bool)


def math_lt(*args: Union[ColT, float]) -> Func:
    """Computes the less than comparison of the column."""
    return math_func("lt", lambda a1, a2: a1 < a2, args, result_type=bool)


def math_le(*args: Union[ColT, float]) -> Func:
    """Computes the less than or equal comparison of the column."""
    return math_func("le", lambda a1, a2: a1 <= a2, args, result_type=bool)


def math_eq(*args: Union[ColT, float]) -> Func:
    """Computes the equality comparison of the column."""
    return math_func("eq", lambda a1, a2: a1 == a2, args, result_type=bool)


def math_ne(*args: Union[ColT, float]) -> Func:
    """Computes the inequality comparison of the column."""
    return math_func("ne", lambda a1, a2: a1 != a2, args, result_type=bool)


def math_gt(*args: Union[ColT, float]) -> Func:
    """Computes the greater than comparison of the column."""
    return math_func("gt", lambda a1, a2: a1 > a2, args, result_type=bool)


def math_ge(*args: Union[ColT, float]) -> Func:
    """Computes the greater than or equal comparison of the column."""
    return math_func("ge", lambda a1, a2: a1 >= a2, args, result_type=bool)
