from collections.abc import Sequence
from typing import Any, Union

from datachain.sql.functions import array

from .func import Func


def cosine_distance(*args: Union[str, Sequence]) -> Func:
    """
    Computes the cosine distance between two vectors.

    The cosine distance is derived from the cosine similarity, which measures the angle
    between two vectors. This function returns the dissimilarity between the vectors,
    where 0 indicates identical vectors and values closer to 1
    indicate higher dissimilarity.

    Args:
        args (str | Sequence): Two vectors to compute the cosine distance between.
            If a string is provided, it is assumed to be the name of the column vector.
            If a sequence is provided, it is assumed to be a vector of values.

    Returns:
        Func: A Func object that represents the cosine_distance function.

    Example:
        ```py
        target_embedding = [0.1, 0.2, 0.3]
        dc.mutate(
            cos_dist1=func.cosine_distance("embedding", target_embedding),
            cos_dist2=func.cosine_distance(target_embedding, [0.4, 0.5, 0.6]),
        )
        ```

    Notes:
        - Ensure both vectors have the same number of elements.
        - Result column will always be of type float.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, str):
            cols.append(arg)
        else:
            func_args.append(list(arg))

    if len(cols) + len(func_args) != 2:
        raise ValueError("cosine_distance() requires exactly two arguments")
    if not cols and len(func_args[0]) != len(func_args[1]):
        raise ValueError("cosine_distance() requires vectors of the same length")

    return Func(
        "cosine_distance",
        inner=array.cosine_distance,
        cols=cols,
        args=func_args,
        result_type=float,
    )


def euclidean_distance(*args: Union[str, Sequence]) -> Func:
    """
    Computes the Euclidean distance between two vectors.

    The Euclidean distance is the straight-line distance between two points
    in Euclidean space. This function returns the distance between the two vectors.

    Args:
        args (str | Sequence): Two vectors to compute the Euclidean distance between.
            If a string is provided, it is assumed to be the name of the column vector.
            If a sequence is provided, it is assumed to be a vector of values.

    Returns:
        Func: A Func object that represents the euclidean_distance function.

    Example:
        ```py
        target_embedding = [0.1, 0.2, 0.3]
        dc.mutate(
            eu_dist1=func.euclidean_distance("embedding", target_embedding),
            eu_dist2=func.euclidean_distance(target_embedding, [0.4, 0.5, 0.6]),
        )
        ```

    Notes:
        - Ensure both vectors have the same number of elements.
        - Result column will always be of type float.
    """
    cols, func_args = [], []
    for arg in args:
        if isinstance(arg, str):
            cols.append(arg)
        else:
            func_args.append(list(arg))

    if len(cols) + len(func_args) != 2:
        raise ValueError("euclidean_distance() requires exactly two arguments")
    if not cols and len(func_args[0]) != len(func_args[1]):
        raise ValueError("euclidean_distance() requires vectors of the same length")

    return Func(
        "euclidean_distance",
        inner=array.euclidean_distance,
        cols=cols,
        args=func_args,
        result_type=float,
    )


def length(arg: Union[str, Sequence, Func]) -> Func:
    """
    Returns the length of the array.

    Args:
        arg (str | Sequence | Func): Array to compute the length of.
            If a string is provided, it is assumed to be the name of the array column.
            If a sequence is provided, it is assumed to be an array of values.
            If a Func is provided, it is assumed to be a function returning an array.

    Returns:
        Func: A Func object that represents the array length function.

    Example:
        ```py
        dc.mutate(
            len1=func.array.length("signal.values"),
            len2=func.array.length([1, 2, 3, 4, 5]),
        )
        ```

    Note:
        - Result column will always be of type int.
    """
    if isinstance(arg, (str, Func)):
        cols = [arg]
        args = None
    else:
        cols = None
        args = [arg]

    return Func("length", inner=array.length, cols=cols, args=args, result_type=int)


def contains(arr: Union[str, Sequence, Func], elem: Any) -> Func:
    """
    Checks whether the `arr` array has the `elem` element.

    Args:
        arr (str | Sequence | Func): Array to check for the element.
            If a string is provided, it is assumed to be the name of the array column.
            If a sequence is provided, it is assumed to be an array of values.
            If a Func is provided, it is assumed to be a function returning an array.
        elem (Any): Element to check for in the array.

    Returns:
        Func: A Func object that represents the contains function. Result of the
            function will be 1 if the element is present in the array, and 0 otherwise.

    Example:
        ```py
        dc.mutate(
            contains1=func.array.contains("signal.values", 3),
            contains2=func.array.contains([1, 2, 3, 4, 5], 7),
        )
        ```
    """

    def inner(arg):
        is_json = type(elem) in [list, dict]
        return array.contains(arg, elem, is_json)

    if isinstance(arr, (str, Func)):
        cols = [arr]
        args = None
    else:
        cols = None
        args = [arr]

    return Func("contains", inner=inner, cols=cols, args=args, result_type=int)


def sip_hash_64(arg: Union[str, Sequence]) -> Func:
    """
    Computes the SipHash-64 hash of the array.

    Args:
        arg (str | Sequence): Array to compute the SipHash-64 hash of.
            If a string is provided, it is assumed to be the name of the array column.
            If a sequence is provided, it is assumed to be an array of values.

    Returns:
        Func: A Func object that represents the sip_hash_64 function.

    Example:
        ```py
        dc.mutate(
            hash1=func.sip_hash_64("signal.values"),
            hash2=func.sip_hash_64([1, 2, 3, 4, 5]),
        )
        ```

    Note:
        - This function is only available for the ClickHouse warehouse.
        - Result column will always be of type int.
    """
    if isinstance(arg, str):
        cols = [arg]
        args = None
    else:
        cols = None
        args = [arg]

    return Func(
        "sip_hash_64", inner=array.sip_hash_64, cols=cols, args=args, result_type=int
    )
