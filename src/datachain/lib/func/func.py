from typing import TYPE_CHECKING, Callable, Optional

from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.utils import DataChainColumnError
from datachain.query.schema import Column, ColumnMeta

if TYPE_CHECKING:
    from datachain import DataType
    from datachain.lib.signal_schema import SignalSchema


class Func:
    def __init__(
        self,
        inner: Callable,
        col: Optional[str] = None,
        result_type: Optional["DataType"] = None,
        is_array: bool = False,
    ) -> None:
        self.inner = inner
        self.col = col
        self.result_type = result_type
        self.is_array = is_array

    @property
    def db_col(self) -> Optional[str]:
        return ColumnMeta.to_db_name(self.col) if self.col else None

    def db_col_type(self, signals_schema: "SignalSchema") -> Optional["DataType"]:
        if not self.db_col:
            return None
        col_type: type = signals_schema.get_column_type(self.db_col)
        return list[col_type] if self.is_array else col_type  # type: ignore[valid-type]

    def get_result_type(self, signals_schema: "SignalSchema") -> "DataType":
        col_type = self.db_col_type(signals_schema)

        if self.result_type:
            return self.result_type

        if col_type:
            return col_type

        raise DataChainColumnError(
            str(self.inner),
            "Column name is required to infer result type",
        )

    def get_column(
        self, signals_schema: "SignalSchema", label: Optional[str] = None
    ) -> Column:
        if self.col:
            if label == "collect":
                print(label)
            col_type = self.get_result_type(signals_schema)
            col = Column(self.db_col, python_to_sql(col_type))
            func_col = self.inner(col)
        else:
            func_col = self.inner()

        if label:
            func_col = func_col.label(label)

        return func_col
