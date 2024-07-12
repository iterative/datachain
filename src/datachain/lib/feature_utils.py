import inspect
import string
from collections.abc import Sequence
from types import GenericAlias
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, create_model
from typing_extensions import Literal as LiteralEx

from datachain.lib.feature import (
    Feature,
    FeatureType,
    FeatureTypeNames,
    convert_type_to_datachain,
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


def _to_feature_type(anno):
    if (
        inspect.isclass(anno)
        and not isinstance(anno, GenericAlias)
        and issubclass(anno, BaseModel)
    ):
        return pydantic_to_feature(anno)

    orig = get_origin(anno)
    args = get_args(anno)
    if args and orig not in (Literal, LiteralEx):
        # recursively get features from each arg
        anno = orig[tuple(_to_feature_type(arg) for arg in args)]

    # check that type can be converted
    convert_type_to_datachain(anno)

    return anno


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
