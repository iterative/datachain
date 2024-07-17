import copy
import inspect
import re
from collections.abc import Sequence
from typing import Any, get_origin

from pydantic import BaseModel

from datachain.query.schema import DEFAULT_DELIMITER


def unflatten_to_json(model: type[BaseModel], row: Sequence[Any], pos=0) -> dict:
    return unflatten_to_json_pos(model, row, pos)[0]


def unflatten_to_json_pos(
    model: type[BaseModel], row: Sequence[Any], pos=0
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
            res[name], pos = unflatten_to_json_pos(anno, row, pos)
        else:
            res[name] = row[pos]
            pos += 1
    return res, pos


def _normalize(name: str) -> str:
    if DEFAULT_DELIMITER in name:
        raise RuntimeError(
            f"variable '{name}' cannot be used "
            f"because it contains {DEFAULT_DELIMITER}"
        )
    return _to_snake_case(name)


def _to_snake_case(name: str) -> str:
    """Convert a CamelCase name to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _unflatten_with_path(model: type[BaseModel], dump, name_path: list[str]):
    res = {}
    for name, f_info in model.model_fields.items():
        anno = f_info.annotation
        name_norm = _normalize(name)
        lst = copy.copy(name_path)

        if inspect.isclass(anno) and issubclass(anno, BaseModel):
            lst.append(name_norm)
            val = _unflatten_with_path(anno, dump, lst)
            res[name] = val
        else:
            lst.append(name_norm)
            curr_path = DEFAULT_DELIMITER.join(lst)
            res[name] = dump[curr_path]
    return model(**res)


def unflatten(model: type[BaseModel], dump):
    return _unflatten_with_path(model, dump, [])
