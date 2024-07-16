import string
from collections.abc import Sequence
from typing import Any, Union

from pydantic import BaseModel, create_model

from datachain.lib.feature import (
    FeatureType,
    FeatureTypeNames,
    is_feature_type,
)
from datachain.lib.utils import DataChainParamsError

AUTO_FEATURE_PREFIX = "_auto_fr"
SUFFIX_SYMBOLS = string.digits + string.ascii_lowercase


class FeatureToTupleError(DataChainParamsError):
    def __init__(self, ds_name, msg):
        if ds_name:
            ds_name = f"' {ds_name}'"
        super().__init__(f"Cannot convert features for dataset{ds_name}: {msg}")


def dict_to_feature(name: str, data_dict: dict[str, FeatureType]) -> type[BaseModel]:
    fields = {name: (anno, ...) for name, anno in data_dict.items()}
    return create_model(name, **fields)  # type: ignore[call-overload]


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
        if not is_feature_type(typ):
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
        output = types_map  # type: ignore[assignment]

    output_types: list[type] = list(output.values())  # type: ignore[union-attr,call-arg,arg-type]
    if len(output) > 1:  # type: ignore[arg-type]
        tuple_type = tuple(output_types)
        res_type = tuple[tuple_type]  # type: ignore[valid-type]
        res_values = list(zip(*fr_map.values()))
    else:
        res_type = output_types[0]  # type: ignore[misc]
        res_values = next(iter(fr_map.values()))

    return res_type, output, res_values
