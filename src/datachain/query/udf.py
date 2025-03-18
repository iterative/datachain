from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

if TYPE_CHECKING:
    from sqlalchemy import Select, Table

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
