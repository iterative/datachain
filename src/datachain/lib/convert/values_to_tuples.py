import itertools
from collections.abc import Sequence
from typing import Any, Union

from datachain.lib.data_model import (
    DataType,
    DataTypeNames,
    DataValue,
    is_chain_type,
)
from datachain.lib.utils import DataChainParamsError


class ValuesToTupleError(DataChainParamsError):
    def __init__(self, ds_name: str, msg: str):
        if ds_name:
            ds_name = f"' {ds_name}'"
        super().__init__(f"Cannot convert signals for dataset{ds_name}: {msg}")


def values_to_tuples(  # noqa: C901, PLR0912
    ds_name: str = "",
    output: Union[None, DataType, Sequence[str], dict[str, DataType]] = None,
    **fr_map: Sequence[DataValue],
) -> tuple[Any, Any, Any]:
    if output:
        if not isinstance(output, (Sequence, str, dict)):
            if len(fr_map) != 1:
                raise ValuesToTupleError(
                    ds_name,
                    f"only one output type was specified, {len(fr_map)} expected",
                )
            if not isinstance(output, type):
                raise ValuesToTupleError(
                    ds_name,
                    f"output must specify a type while '{output}' was given",
                )

            key: str = next(iter(fr_map.keys()))
            output = {key: output}  # type: ignore[dict-item]

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
                f" number of signals '{len(fr_map)}'",
            )

    types_map: dict[str, type] = {}
    length = -1
    for k, v in fr_map.items():
        if not isinstance(v, Sequence) or isinstance(v, str):  # type: ignore[unreachable]
            raise ValuesToTupleError(ds_name, f"signals '{k}' is not a sequence")
        len_ = len(v)

        if output:
            if k not in output:  # type: ignore[operator]
                raise ValuesToTupleError(
                    ds_name,
                    f"signal '{k}' is not present in the output",
                )
        else:
            # FIXME: Stops as soon as it finds the first non-None value.
            # If a non-None value appears early, it won't check the remaining items for
            # `None` values.
            try:
                first_not_none_element = next(
                    itertools.dropwhile(lambda i: i is None, v)
                )
            except StopIteration:
                # set default type to `str` if column is empty or all values are `None`
                typ = str
            else:
                typ = type(first_not_none_element)  # type: ignore[assignment]
                if not is_chain_type(typ):
                    raise ValuesToTupleError(
                        ds_name,
                        f"signal '{k}' has unsupported type '{typ.__name__}'."
                        f" Please use DataModel types: {DataTypeNames}",
                    )
                if isinstance(first_not_none_element, list):
                    typ = list[type(first_not_none_element[0])]  # type: ignore[assignment, misc]
            types_map[k] = typ

        if length < 0:
            length = len_
        elif length != len_:
            raise ValuesToTupleError(
                ds_name,
                f"signal '{k}' should have length {length} while {len_} is given",
            )

    if not output:
        output = types_map  # type: ignore[assignment]

    if not output:
        raise ValuesToTupleError(
            ds_name,
            "output type must be dict[str, DataType] while empty is given"
            " and no signals are provided",
        )

    output_types: list[type] = list(output.values())  # type: ignore[union-attr,call-arg,arg-type]
    if len(output) > 1:  # type: ignore[arg-type]
        tuple_type = tuple(output_types)
        res_type = tuple[tuple_type]  # type: ignore[valid-type]
        res_values: Sequence[Any] = list(zip(*fr_map.values()))
    else:
        res_type = output_types[0]  # type: ignore[misc]
        res_values = next(iter(fr_map.values()))

    return res_type, output, res_values
