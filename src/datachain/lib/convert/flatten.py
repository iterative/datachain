from datetime import datetime

from pydantic import BaseModel

from datachain.lib.model_store import ModelStore
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int,
    Int32,
    Int64,
    NullType,
    String,
)

DATACHAIN_TO_TYPE = {
    Int: int,
    Int32: int,
    Int64: int,
    String: str,
    Float: float,
    Boolean: bool,
    DateTime: datetime,
    Binary: bytes,
    Array(NullType): list,
    JSON: dict,
}


def flatten(obj: BaseModel):
    return tuple(_flatten_fields_values(obj.model_fields, obj))


def flatten_list(obj_list):
    return tuple(
        val for obj in obj_list for val in _flatten_fields_values(obj.model_fields, obj)
    )


def _flatten_fields_values(fields, obj: BaseModel):
    for name, f_info in fields.items():
        anno = f_info.annotation
        # Optimization: Access attributes directly to skip the model_dump() call.
        value = getattr(obj, name)

        if isinstance(value, list):
            if value and ModelStore.is_pydantic(type(value[0])):
                yield [val.model_dump() for val in value]
            else:
                yield value
        elif isinstance(value, dict):
            yield {
                key: val.model_dump() if ModelStore.is_pydantic(type(val)) else val
                for key, val in value.items()
            }
        elif ModelStore.is_pydantic(anno):
            yield from _flatten_fields_values(anno.model_fields, value)
        else:
            yield value


def _flatten(obj):
    return tuple(_flatten_fields_values(obj.model_fields, obj))
