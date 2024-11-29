from typing import Union

from datachain.sql.functions import conditional

from .func import ColT, Func


def greatest(*args: Union[ColT, float]) -> Func:
    """
    Returns the greatest (largest) value from the given input values.

    Args:
        args (ColT | str | int | float | Sequence): The values to compare.
            If a string is provided, it is assumed to be the name of the column.
            If a Func is provided, it is assumed to be a function returning a value.
            If an int, float, or Sequence is provided, it is assumed to be a literal.

    Returns:
        Func: A Func object that represents the greatest function.

    Example:
        ```py
        dc.mutate(
            greatest=func.greatest("signal.value", 0),
        )
        ```

    Note:
        - Result column will always be of the same type as the input columns.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func(
        "greatest",
        inner=conditional.greatest,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def least(*args: Union[ColT, float]) -> Func:
    """
    Returns the least (smallest) value from the given input values.

    Args:
        args (ColT | str | int | float | Sequence): The values to compare.
            If a string is provided, it is assumed to be the name of the column.
            If a Func is provided, it is assumed to be a function returning a value.
            If an int, float, or Sequence is provided, it is assumed to be a literal.

    Returns:
        Func: A Func object that represents the least function.

    Example:
        ```py
        dc.mutate(
            least=func.least("signal.value", 0),
        )
        ```

    Note:
        - Result column will always be of the same type as the input columns.
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func(
        "least", inner=conditional.least, cols=cols, args=func_args, result_type=int
    )
