from typing import Union

from datachain.sql.functions import numeric

from .func import ColT, Func


def bit_and(*args: Union[ColT, int]) -> Func:
    """
    Computes the bitwise AND operation between two values.

    Args:
        args (str | int): Two values to compute the bitwise AND operation between.
            If a string is provided, it is assumed to be the name of the column vector.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A Func object that represents the bitwise AND function.

    Example:
        ```py
        dc.mutate(
            xor1=func.bit_and("signal.values", 0x0F),
        )
        ```

    Notes:
        - Result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_and() requires exactly two arguments")

    return Func(
        "bit_and",
        inner=numeric.bit_and,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def bit_or(*args: Union[ColT, int]) -> Func:
    """
    Computes the bitwise AND operation between two values.

    Args:
        args (str | int): Two values to compute the bitwise OR operation between.
            If a string is provided, it is assumed to be the name of the column vector.
            If an integer is provided, it is assumed to be a constant value.

    Returns:
        Func: A Func object that represents the bitwise OR function.

    Example:
        ```py
        dc.mutate(
            xor1=func.bit_or("signal.values", 0x0F),
        )
        ```

    Notes:
        - Result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_or() requires exactly two arguments")

    return Func(
        "bit_or",
        inner=numeric.bit_or,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def bit_xor(*args: Union[ColT, int]) -> Func:
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
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_xor() requires exactly two arguments")

    return Func(
        "bit_xor",
        inner=numeric.bit_xor,
        cols=cols,
        args=func_args,
        result_type=int,
    )


def int_hash_64(col: Union[ColT, int]) -> Func:
    """
    Returns the 64-bit hash of an integer.

    Args:
        col (str | int): String to compute the hash of.
            If a string is provided, it is assumed to be the name of the column.
            If a int is provided, it is assumed to be an int literal.
            If a Func is provided, it is assumed to be a function returning an int.

    Returns:
        Func: A Func object that represents the 64-bit hash function.

    Example:
        ```py
        dc.mutate(
            val_hash=func.int_hash_64("val"),
        )
        ```

    Note:
        - Result column will always be of type int.
    """
    cols, args = [], []
    if isinstance(col, int):
        args.append(col)
    else:
        cols.append(col)

    return Func(
        "int_hash_64", inner=numeric.int_hash_64, cols=cols, args=args, result_type=int
    )


def bit_hamming_distance(*args: Union[ColT, int]) -> Func:
    """
    Computes the Hamming distance between the bit representations of two integer values.

    The Hamming distance is the number of positions at which the corresponding bits
    are different. This function returns the dissimilarity between the integers,
    where 0 indicates identical integers and values closer to the number of bits
    in the integer indicate higher dissimilarity.

    Args:
        args (str | int): Two integers to compute the Hamming distance between.
            If a str is provided, it is assumed to be the name of the column.
            If an int is provided, it is assumed to be an integer literal.

    Returns:
        Func: A Func object that represents the Hamming distance function.

    Example:
        ```py
        dc.mutate(
            ham_dist=func.bit_hamming_distance("embed1", 123456),
        )
        ```

    Notes:
        - Result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, int):
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("bit_hamming_distance() requires exactly two arguments")

    return Func(
        "bit_hamming_distance",
        inner=numeric.bit_hamming_distance,
        cols=cols,
        args=func_args,
        result_type=int,
    )
