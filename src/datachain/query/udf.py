from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, Union

if TYPE_CHECKING:
    from sqlalchemy import Select, Table

    from datachain.catalog import Catalog
    from datachain.query.batch import BatchingStrategy


class UdfInfo(TypedDict):
    udf_data: bytes
    catalog_init: dict[str, Any]
    metastore_clone_params: tuple[Callable[..., Any], list[Any], dict[str, Any]]
    warehouse_clone_params: tuple[Callable[..., Any], list[Any], dict[str, Any]]
    table: "Table"
    query: "Select"
    udf_fields: list[str]
    batching: "BatchingStrategy"
    processes: Optional[int]
    is_generator: bool
    cache: bool
    rows_total: int


class AbstractUDFDistributor(ABC):
    @abstractmethod
    def __init__(
        self,
        catalog: "Catalog",
        table: "Table",
        query: "Select",
        udf_data: bytes,
        batching: "BatchingStrategy",
        workers: Union[bool, int],
        processes: Union[bool, int],
        udf_fields: list[str],
        rows_total: int,
        use_cache: bool,
        is_generator: bool = False,
        min_task_size: Optional[Union[str, int]] = None,
    ) -> None: ...

    @abstractmethod
    def __call__(self) -> None: ...

    @staticmethod
    @abstractmethod
    def run_udf(fd: Optional[int] = None) -> int: ...
