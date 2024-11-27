from datachain.sql.functions import path

from .func import ColT, Func


def parent(col: ColT) -> Func:
    """
    Returns the directory component of a posix-style path.

    Args:
        col (str | literal | Func): String to compute the path parent of.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.

    Returns:
        Func: A Func object that represents the path parent function.

    Example:
        ```py
        dc.mutate(
            parent=func.path.parent("file.path"),
        )
        ```

    Note:
        - Result column will always be of type string.
    """
    return Func("parent", inner=path.parent, cols=[col], result_type=str)


def name(col: ColT) -> Func:
    """
    Returns the final component of a posix-style path.

    Args:
        col (str | literal): String to compute the path name of.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.

    Returns:
        Func: A Func object that represents the path name function.

    Example:
        ```py
        dc.mutate(
            file_name=func.path.name("file.path"),
        )
        ```

    Note:
        - Result column will always be of type string.
    """

    return Func("name", inner=path.name, cols=[col], result_type=str)


def file_stem(col: ColT) -> Func:
    """
    Returns the path without the extension.

    Args:
        col (str | literal): String to compute the file stem of.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.

    Returns:
        Func: A Func object that represents the file stem function.

    Example:
        ```py
        dc.mutate(
            file_stem=func.path.file_stem("file.path"),
        )
        ```

    Note:
        - Result column will always be of type string.
    """

    return Func("file_stem", inner=path.file_stem, cols=[col], result_type=str)


def file_ext(col: ColT) -> Func:
    """
    Returns the extension of the given path.

    Args:
        col (str | literal): String to compute the file extension of.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.
            If a Func is provided, it is assumed to be a function returning a string.

    Returns:
        Func: A Func object that represents the file extension function.

    Example:
        ```py
        dc.mutate(
            file_stem=func.path.file_ext("file.path"),
        )
        ```

    Note:
        - Result column will always be of type string.
    """

    return Func("file_ext", inner=path.file_ext, cols=[col], result_type=str)
