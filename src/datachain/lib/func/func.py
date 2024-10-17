from typing import TYPE_CHECKING, Callable, Optional

from sqlalchemy import desc

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import DataChainColumnError
from datachain.query.schema import Column, ColumnMeta

if TYPE_CHECKING:
    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema


class Window:
    def __init__(self, *, partition_by: str, order_by: str, desc: bool = False) -> None:
        self.partition_by = partition_by
        self.order_by = order_by
        self.desc = desc


class Func:
    def __init__(
        self,
        inner: Callable,
        col: Optional[str] = None,
        result_type: Optional["DataType"] = None,
        is_array: bool = False,
        is_window: bool = False,
        window: Optional[Window] = None,
    ) -> None:
        self.inner = inner
        self.col = col
        self.result_type = result_type
        self.is_array = is_array
        self.is_window = is_window
        self.window = window

    def over(self, window: Window) -> "Func":
        if not self.is_window:
            raise DataChainColumnError(
                str(self.inner), "Window function is not supported"
            )

        return Func(
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
            str(self.inner),
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
                raise DataChainColumnError(
                    str(self.inner), "Window function requires window"
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
