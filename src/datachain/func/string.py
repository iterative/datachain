from typing import Optional, Union, get_origin

from sqlalchemy import literal

from datachain.sql.functions import string

from .func import Func


def length(col: Union[str, Func]) -> Func:
    """
    Returns the length of the string.

    Args:
        col (str | literal | Func): String to compute the length of.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.

    Returns:
        Func: A Func object that represents the string length function.

    Example:
        ```py
        dc.mutate(
            len1=func.string.length("file.path"),
            len2=func.string.length("Random string"),
        )
        ```

    Note:
        - Result column will always be of type int.
    """
    return Func("length", inner=string.length, cols=[col], result_type=int)


def split(col: Union[str, Func], sep: str, limit: Optional[int] = None) -> Func:
    """
    Takes a column and split character and returns an array of the parts.

    Args:
        col (str | literal): Column to split.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.
        sep (str): Separator to split the string.
        limit (int, optional): Maximum number of splits to perform.

    Returns:
        Func: A Func object that represents the split function.

    Example:
        ```py
        dc.mutate(
            path_parts=func.string.split("file.path", "/"),
            str_words=func.string.length("Random string", " "),
        )
        ```

    Note:
        - Result column will always be of type array of strings.
    """

    def inner(arg):
        if limit is not None:
            return string.split(arg, sep, limit)
        return string.split(arg, sep)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("split", inner=inner, cols=cols, args=args, result_type=list[str])


def replace(col: Union[str, Func], pattern: str, replacement: str) -> Func:
    """
    Replaces substring with another string.

    Args:
        col (str | literal): Column to split.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.
        pattern (str): Pattern to replace.
        replacement (str): Replacement string.

    Returns:
        Func: A Func object that represents the replace function.

    Example:
        ```py
        dc.mutate(
            signal=func.string.replace("signal.name", "pattern", "replacement),
        )
        ```

    Note:
        - Result column will always be of type string.
    """

    def inner(arg):
        return string.replace(arg, pattern, replacement)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("replace", inner=inner, cols=cols, args=args, result_type=str)


def regexp_replace(col: Union[str, Func], regex: str, replacement: str) -> Func:
    r"""
    Replaces substring that match a regular expression.

    Args:
        col (str | literal): Column to split.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.
        regex (str): Regular expression pattern to replace.
        replacement (str): Replacement string.

    Returns:
        Func: A Func object that represents the regexp_replace function.

    Example:
        ```py
        dc.mutate(
            signal=func.string.regexp_replace("signal.name", r"\d+", "X"),
        )
        ```

    Note:
        - Result column will always be of type string.
    """

    def inner(arg):
        return string.regexp_replace(arg, regex, replacement)

    if get_origin(col) is literal:
        cols = None
        args = [col]
    else:
        cols = [col]
        args = None

    return Func("regexp_replace", inner=inner, cols=cols, args=args, result_type=str)


def byte_hamming_distance(*args: Union[str, Func]) -> Func:
    """
    Computes the Hamming distance between two strings.

    The Hamming distance is the number of positions at which the corresponding
    characters are different. This function returns the dissimilarity between
    the strings, where 0 indicates identical strings and values closer to the length
    of the strings indicate higher dissimilarity.

    Args:
        args (str | literal): Two strings to compute the Hamming distance between.
            If a str is provided, it is assumed to be the name of the column.
            If a Literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A Func object that represents the Hamming distance function.

    Example:
        ```py
        dc.mutate(
            ham_dist=func.byte_hamming_distance("file.phash", literal("hello")),
        )
        ```

    Notes:
        - Result column will always be of type int.
    """
    cols, func_args = [], []
    for arg in args:
        if get_origin(arg) is literal:
            func_args.append(arg)
        else:
            cols.append(arg)

    if len(cols) + len(func_args) != 2:
        raise ValueError("byte_hamming_distance() requires exactly two arguments")

    return Func(
        "byte_hamming_distance",
        inner=string.byte_hamming_distance,
        cols=cols,
        args=func_args,
        result_type=int,
    )
