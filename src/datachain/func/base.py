from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain.lib.signal_schema import SignalSchema
    from datachain.query.schema import Column


class Function:
    __metaclass__ = ABCMeta

    name: str

    @abstractmethod
    def get_column(
        self,
        signals_schema: Optional["SignalSchema"] = None,
        label: Optional[str] = None,
        table: Optional["TableClause"] = None,
    ) -> "Column":
        pass
