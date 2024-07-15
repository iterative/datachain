import copy
import inspect
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from typing import (
    Any,
    get_origin,
)

from pydantic import BaseModel

from datachain.lib.model_store import ModelStore
from datachain.query.schema import DEFAULT_DELIMITER
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


# Disable Pydantic warning, see https://github.com/iterative/dvcx/issues/1285
warnings.filterwarnings(
    "ignore",
    message="Field name .* shadows an attribute in parent",
    category=UserWarning,
)


class ModelUtil:
    @classmethod
    def flatten(cls, obj: BaseModel):
        return tuple(cls._flatten_fields_values(obj.model_fields, obj))

    @classmethod
    def _flatten_fields_values(cls, fields, obj: BaseModel):
        for name, f_info in fields.items():
            anno = f_info.annotation
            # Optimization: Access attributes directly to skip the model_dump() call.
            value = getattr(obj, name)

            if isinstance(value, list):
                yield [
                    val.model_dump() if ModelStore.is_pydantic(type(val)) else val
                    for val in value
                ]
            elif isinstance(value, dict):
                yield {
                    key: val.model_dump() if ModelStore.is_pydantic(type(val)) else val
                    for key, val in value.items()
                }
            elif ModelStore.is_pydantic(anno):
                yield from cls._flatten_fields_values(anno.model_fields, value)
            else:
                yield value

    @classmethod
    def unflatten_to_json(
        cls, model: type[BaseModel], row: Sequence[Any], pos=0
    ) -> dict:
        return cls.unflatten_to_json_pos(model, row, pos)[0]

    @classmethod
    def unflatten_to_json_pos(
        cls, model: type[BaseModel], row: Sequence[Any], pos=0
    ) -> tuple[dict, int]:
        res = {}
        for name, f_info in model.model_fields.items():
            anno = f_info.annotation
            origin = get_origin(anno)
            if (
                origin not in (list, dict)
                and inspect.isclass(anno)
                and issubclass(anno, BaseModel)
            ):
                res[name], pos = cls.unflatten_to_json_pos(anno, row, pos)
            else:
                res[name] = row[pos]
                pos += 1
        return res, pos

    #################
    @classmethod
    def _flatten(cls, obj):
        return tuple(cls._flatten_fields_values(obj.model_fields, obj))

    @classmethod
    def flatten_list(cls, obj_list):
        return tuple(
            val
            for obj in obj_list
            for val in cls._flatten_fields_values(obj.model_fields, obj)
        )

    ##################  UNFLATTEN:
    @classmethod
    def _normalize(cls, name: str) -> str:
        if DEFAULT_DELIMITER in name:
            raise RuntimeError(
                f"variable '{name}' cannot be used "
                f"because it contains {DEFAULT_DELIMITER}"
            )
        return cls._to_snake_case(name)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a CamelCase name to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @classmethod
    def _unflatten_with_path(cls, model: type[BaseModel], dump, name_path: list[str]):
        res = {}
        for name, f_info in model.model_fields.items():
            anno = f_info.annotation
            name_norm = cls._normalize(name)
            lst = copy.copy(name_path)

            if inspect.isclass(anno) and issubclass(anno, BaseModel):
                lst.append(name_norm)
                val = cls._unflatten_with_path(anno, dump, lst)
                res[name] = val
            else:
                lst.append(name_norm)
                curr_path = DEFAULT_DELIMITER.join(lst)
                res[name] = dump[curr_path]
        return model(**res)

    @classmethod
    def unflatten(cls, model: type[BaseModel], dump):
        return cls._unflatten_with_path(model, dump, [])
