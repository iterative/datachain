from collections.abc import Generator

from pydantic import BaseModel

from datachain.lib.model_store import ModelStore


def flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(obj.model_fields, obj))


def flatten_list(obj_list: list[BaseModel]) -> tuple:
    return tuple(
        val for obj in obj_list for val in _flatten_fields_values(obj.model_fields, obj)
    )


def _flatten_list_field(value: list) -> list:
    assert isinstance(value, list)
    if value and ModelStore.is_pydantic(type(value[0])):
        return [val.model_dump() for val in value]
    if value and isinstance(value[0], list):
        return [_flatten_list_field(v) for v in value]
    return value


def _flatten_fields_values(fields: dict, obj: BaseModel) -> Generator:
    for name, f_info in fields.items():
        anno = f_info.annotation
        # Optimization: Access attributes directly to skip the model_dump() call.
        value = getattr(obj, name)
        if isinstance(value, list):
            yield _flatten_list_field(value)
        elif isinstance(value, dict):
            yield {
                key: val.model_dump() if ModelStore.is_pydantic(type(val)) else val
                for key, val in value.items()
            }
        elif ModelStore.is_pydantic(anno):
            yield from _flatten_fields_values(anno.model_fields, value)
        else:
            yield value


def _flatten(obj: BaseModel) -> tuple:
    return tuple(_flatten_fields_values(obj.model_fields, obj))
