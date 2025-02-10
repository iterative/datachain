from typing import Optional, Union

from sqlalchemy import ColumnElement
from sqlalchemy import and_ as sql_and
from sqlalchemy import case as sql_case
from sqlalchemy import or_ as sql_or

from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import Column
from datachain.sql.functions import conditional

from .func import ColT, Func

CaseT = Union[int, float, complex, bool, str, Func, ColumnElement]


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


def case(
    *args: tuple[Union[ColumnElement, Func, bool], CaseT], else_: Optional[CaseT] = None
) -> Func:
    """
    Returns the case function that produces case expression which has a list of
    conditions and corresponding results. Results can be python primitives like string,
    numbers or booleans but can also be other nested functions (including case function)
    or columns.
    Result type is inferred from condition results.

    Args:
        args tuple((ColumnElement | Func | bool),(str | int | float | complex | bool, Func, ColumnElement)):
            Tuple of condition and values pair.
        else_ (str | int | float | complex | bool, Func): optional else value in case
            expression. If omitted, and no case conditions are satisfied, the result
            will be None (NULL in DB).

    Returns:
        Func: A Func object that represents the case function.

    Example:
        ```py
        dc.mutate(
            res=func.case((C("num") > 0, "P"), (C("num") < 0, "N"), else_="Z"),
        )
        ```
    """  # noqa: E501
    supported_types = [int, float, complex, str, bool]

    def _get_type(val):
        from enum import Enum

        if isinstance(val, Func):
            # nested functions
            return val.result_type
        if isinstance(val, Column):
            # at this point we cannot know what is the type of a column
            return None
        if isinstance(val, Enum):
            return type(val.value)
        return type(val)

    if not args:
        raise DataChainParamsError("Missing statements")

    type_ = _get_type(else_) if else_ is not None else None

    for arg in args:
        arg_type = _get_type(arg[1])
        if arg_type is None:
            # we couldn't figure out the type of case value
            continue
        if type_ and arg_type != type_:
            raise DataChainParamsError(
                f"Statement values must be of the same type, got {type_} and {arg_type}"
            )
        type_ = arg_type

    if type_ is not None and type_ not in supported_types:
        raise DataChainParamsError(
            f"Only python literals ({supported_types}) are supported for values"
        )

    kwargs = {"else_": else_}

    return Func("case", inner=sql_case, cols=args, kwargs=kwargs, result_type=type_)


def ifelse(
    condition: Union[ColumnElement, Func], if_val: CaseT, else_val: CaseT
) -> Func:
    """
    Returns the ifelse function that produces if expression which has a condition
    and values for true and false outcome. Results can be one of python primitives
    like string, numbers or booleans, but can also be nested functions or columns.
    Result type is inferred from the values.

    Args:
        condition (ColumnElement, Func):  Condition which is evaluated.
        if_val (str | int | float | complex | bool, Func, ColumnElement): Value for true
            condition outcome.
        else_val (str | int | float | complex | bool, Func, ColumnElement): Value for
            false condition outcome.

    Returns:
        Func: A Func object that represents the ifelse function.

    Example:
        ```py
        dc.mutate(
            res=func.ifelse(isnone("col"), "EMPTY", "NOT_EMPTY")
        )
        ```
    """
    return case((condition, if_val), else_=else_val)


def isnone(col: Union[str, Column]) -> Func:
    """
    Returns True if column value is None, otherwise False.

    Args:
        col (str | Column): Column to check if it's None or not.
            If a string is provided, it is assumed to be the name of the column.

    Returns:
        Func: A Func object that represents the conditional to check if column is None.

    Example:
        ```py
        dc.mutate(test=ifelse(isnone("col"), "EMPTY", "NOT_EMPTY"))
        ```
    """
    from datachain import C

    if isinstance(col, str):
        # if string, it is assumed to be the name of the column
        col = C(col)

    return case((col.is_(None) if col is not None else True, True), else_=False)


def or_(*args: Union[ColumnElement, Func]) -> Func:
    """
    Returns the function that produces conjunction of expressions joined by OR
    logical operator.

    Args:
        args (ColumnElement | Func): The expressions for OR statement.

    Returns:
        Func: A Func object that represents the or function.

    Example:
        ```py
        dc.mutate(
            test=ifelse(or_(isnone("name"), C("name") == ''), "Empty", "Not Empty")
        )
        ```
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func("or", inner=sql_or, cols=cols, args=func_args, result_type=bool)


def and_(*args: Union[ColumnElement, Func]) -> Func:
    """
    Returns the function that produces conjunction of expressions joined by AND
    logical operator.

    Args:
        args (ColumnElement | Func): The expressions for AND statement.

    Returns:
        Func: A Func object that represents the and function.

    Example:
        ```py
        dc.mutate(
            test=ifelse(and_(isnone("name"), isnone("surname")), "Empty", "Not Empty")
        )
        ```
    """
    cols, func_args = [], []

    for arg in args:
        if isinstance(arg, (str, Func)):
            cols.append(arg)
        else:
            func_args.append(arg)

    return Func("and", inner=sql_and, cols=cols, args=func_args, result_type=bool)
