from typing import Union

from datachain.sql.functions import numeric

from .func import Func


def bit_xor(*args: Union[str, int]) -> Func:
    """
    Computes the bitwise XOR operation between two values.

    Args:
        args (str | int): Two values to compute the bitwise XOR operation between.
            If a string is provided, it is assumed to be the name of the column vector.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A Func object that represents the bitwise XOR function.

    Example:
        ```py
        dc.mutate(
            xor1=func.bit_xor("signal.values", 0x0F),
        )
        ```

    Notes:
        - Result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, str):
            cols.append(arg)
        else:
            func_args.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_xor() requires exactly two arguments")

    return Func(
        "bit_xor",
        inner=numeric.bit_xor,
        cols=cols,
        args=func_args,
        result_type=int,
    )
