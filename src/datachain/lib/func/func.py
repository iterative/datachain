from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy import desc

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import DataChainColumnError, DataChainParamsError
from datachain.query.schema import Column, ColumnMeta

if TYPE_CHECKING:
    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema


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


class Func:
    """Represents a function to be applied to a column in a SQL query."""

    def __init__(
        self,
        name: str,
        inner: Callable,
        col: Optional[str] = None,
        result_type: Optional["DataType"] = None,
        is_array: bool = False,
        is_window: bool = False,
        window: Optional[Window] = None,
    ) -> None:
        self.name = name
        self.inner = inner
        self.col = col
        self.result_type = result_type
        self.is_array = is_array
        self.is_window = is_window
        self.window = window

    def __str__(self) -> str:
        return self.name + "()"

    def over(self, window: Window) -> "Func":
        if not self.is_window:
            raise DataChainParamsError(f"{self} doesn't support window (over())")

        return Func(
            "over",
            self.inner,
            self.col,
            self.result_type,
            self.is_array,
            self.is_window,
            window,
        )

    @property
    def db_col(self) -> Optional[str]:
        return ColumnMeta.to_db_name(self.col) if self.col else None

    def db_col_type(self, signals_schema: "SignalSchema") -> Optional["DataType"]:
        if not self.db_col:
            return None
        col_type: type = signals_schema.get_column_type(self.db_col)
        return list[col_type] if self.is_array else col_type  # type: ignore[valid-type]

    def get_result_type(self, signals_schema: "SignalSchema") -> "DataType":
        if self.result_type:
            return self.result_type

        if col_type := self.db_col_type(signals_schema):
            return col_type

        raise DataChainColumnError(
            str(self),
            "Column name is required to infer result type",
        )

    def get_column(
        self, signals_schema: "SignalSchema", label: Optional[str] = None
    ) -> Column:
        col_type = self.get_result_type(signals_schema)
        sql_type = python_to_sql(col_type)

        if self.col:
            col = Column(self.db_col, sql_type)
            func_col = self.inner(col)
        else:
            func_col = self.inner()

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

        func_col.type = sql_type

        if label:
            func_col = func_col.label(label)

        return func_col
