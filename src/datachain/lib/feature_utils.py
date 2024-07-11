import inspect
import string
from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel, create_model
from typing_extensions import Literal as LiteralEx

from datachain.lib.feature import (
    TYPE_TO_DATACHAIN,
    Feature,
    FeatureType,
    FeatureTypeNames,
)
from datachain.lib.utils import DataChainParamsError

AUTO_FEATURE_PREFIX = "_auto_fr"
SUFFIX_SYMBOLS = string.digits + string.ascii_lowercase


class FeatureToTupleError(DataChainParamsError):
    def __init__(self, ds_name, msg):
        if ds_name:
            ds_name = f"' {ds_name}'"
        super().__init__(f"Cannot convert features for dataset{ds_name}: {msg}")


feature_cache: dict[type[BaseModel], type[Feature]] = {}


def pydantic_to_feature(data_cls: type[BaseModel]) -> type[Feature]:
    if data_cls in feature_cache:
        return feature_cache[data_cls]

    fields = {}
    for name, field_info in data_cls.model_fields.items():
        anno = _to_feature_type(field_info.annotation)
        fields[name] = (anno, field_info.default)

    cls = create_model(
        data_cls.__name__,
        __base__=(data_cls, Feature),  # type: ignore[call-overload]
        **fields,
    )
    feature_cache[data_cls] = cls
    return cls


def _to_feature_type(anno):  # noqa: PLR0911
    if anno in feature_cache:
        return feature_cache[anno]

    if anno in TYPE_TO_DATACHAIN:
        return anno

    if anno is type(None):
        return type(None)

    orig = get_origin(anno)
    args = get_args(anno)

    if orig in (Literal, LiteralEx):
        return str

    if orig is Optional:
        return Optional[_to_feature_type(args[0])]

    if orig is Annotated:
        # Ignoring annotations
        return _to_feature_type(args[0])

    if orig is list:
        if len(args) > 1:
            raise TypeError(
                "type conversion error: list is suppose to have only 1 value"
            )
        return list[_to_feature_type(args[0])]

    if orig == Union:
        vals = [_to_feature_type(arg) for arg in args]
        return Union[tuple(vals)]

    if inspect.isclass(anno):
        if issubclass(anno, BaseModel):
            return pydantic_to_feature(anno)
        if issubclass(anno, Enum):
            return str
        if anno is object:
            return object

    raise TypeError(f"Cannot recognize type {anno}")


def features_to_tuples(
    ds_name: str = "",
    output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
    **fr_map,
) -> tuple[Any, Any, Any]:
    types_map = {}
    length = -1
    for k, v in fr_map.items():
        if not isinstance(v, Sequence) or isinstance(v, str):
            raise FeatureToTupleError(ds_name, f"features '{k}' is not a sequence")
        len_ = len(v)

        if len_ == 0:
            raise FeatureToTupleError(ds_name, f"feature '{k}' is empty list")

        if length < 0:
            length = len_
        elif length != len_:
            raise FeatureToTupleError(
                ds_name,
                f"feature '{k}' should have length {length} while {len_} is given",
            )
        typ = type(v[0])
        if not Feature.is_feature_type(typ):
            raise FeatureToTupleError(
                ds_name,
                f"feature '{k}' has unsupported type '{typ.__name__}'."
                f" Please use Feature types: {FeatureTypeNames}",
            )
        types_map[k] = typ
    if output:
        if not isinstance(output, Sequence) and not isinstance(output, str):
            if len(fr_map) != 1:
                raise FeatureToTupleError(
                    ds_name,
                    f"only one output type was specified, {len(fr_map)} expected",
                )
            if not isinstance(output, type):
                raise FeatureToTupleError(
                    ds_name,
                    f"output must specify a type while '{output}' was given",
                )

            key: str = next(iter(fr_map.keys()))
            output = {key: output}  # type: ignore[dict-item]

        if len(output) != len(fr_map):
            raise FeatureToTupleError(
                ds_name,
                f"number of outputs '{len(output)}' should match"
                f" number of features '{len(fr_map)}'",
            )
        if isinstance(output, dict):
            raise FeatureToTupleError(
                ds_name,
                "output type must be dict[str, FeatureType] while "
                f"'{type(output).__name__}' is given",
            )
    else:
        output = types_map

    output_types: list[type] = list(output.values())  # type: ignore[union-attr,arg-type]
    if len(output) > 1:
        tuple_type = tuple(output_types)
        res_type = tuple[tuple_type]  # type: ignore[valid-type]
        res_values = list(zip(*fr_map.values()))
    else:
        res_type = output_types[0]  # type: ignore[misc]
        res_values = next(iter(fr_map.values()))

    return res_type, output, res_values
