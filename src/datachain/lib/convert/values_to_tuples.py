from collections.abc import Sequence
from typing import Any, Union

from datachain.lib.data_model import DataType, DataTypeNames, is_chain_type
from datachain.lib.utils import DataChainParamsError


class ValuesToTupleError(DataChainParamsError):
    def __init__(self, ds_name, msg):
        if ds_name:
            ds_name = f"' {ds_name}'"
        super().__init__(f"Cannot convert features for dataset{ds_name}: {msg}")


def values_to_tuples(
    ds_name: str = "",
    output: Union[None, dict[str, DataType]] = None,
    **fr_map,
) -> tuple[Any, Any, Any]:
    if output:
        if not isinstance(output, dict):
            raise ValuesToTupleError(
                ds_name,
                "output type must be dict[str, DataType] while "
                f"'{type(output).__name__}' is given",
            )

        if len(output) != len(fr_map):
            raise ValuesToTupleError(
                ds_name,
                f"number of outputs '{len(output)}' should match"
                f" number of features '{len(fr_map)}'",
            )

    types_map = {}
    length = -1
    for k, v in fr_map.items():
        if not isinstance(v, Sequence) or isinstance(v, str):
            raise ValuesToTupleError(ds_name, f"features '{k}' is not a sequence")
        len_ = len(v)

        if output:
            if k not in output:
                raise ValuesToTupleError(
                    ds_name,
                    f"feature '{k}' is not present in the output",
                )
        else:
            if len_ == 0:
                raise ValuesToTupleError(ds_name, f"feature '{k}' is empty list")

            typ = type(v[0])
            if not is_chain_type(typ):
                raise ValuesToTupleError(
                    ds_name,
                    f"feature '{k}' has unsupported type '{typ.__name__}'."
                    f" Please use DataModel types: {DataTypeNames}",
                )
            types_map[k] = typ

        if length < 0:
            length = len_
        elif length != len_:
            raise ValuesToTupleError(
                ds_name,
                f"feature '{k}' should have length {length} while {len_} is given",
            )

    if not output:
        output = types_map  # type: ignore[assignment]

    if not output:
        raise ValuesToTupleError(
            ds_name,
            "output type must be dict[str, DataType] while empty is given"
            " and no features are provided",
        )

    output_types: list[type] = list(output.values())  # type: ignore[union-attr,call-arg,arg-type]
    if len(output) > 1:  # type: ignore[arg-type]
        tuple_type = tuple(output_types)
        res_type = tuple[tuple_type]  # type: ignore[valid-type]
        res_values = list(zip(*fr_map.values()))
    else:
        res_type = output_types[0]  # type: ignore[misc]
        res_values = next(iter(fr_map.values()))

    return res_type, output, res_values
