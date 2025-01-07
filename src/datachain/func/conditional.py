from typing import Union

from sqlalchemy import case as sql_case
from sqlalchemy.sql.elements import BinaryExpression

from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import Column
from datachain.sql.functions import conditional

from .func import ColT, Func

CaseT = Union[int, float, complex, bool, str]


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


def case(*args: tuple[BinaryExpression, CaseT], else_=None) -> Func:
    """
    Returns the case function that produces case expression which has a list of
    conditions and corresponding results. Results can only be python primitives
    like string, numbes or booleans. Result type is inferred from condition results.

    Args:
        args (tuple(BinaryExpression, value(str | int | float | complex | bool):
            - Tuple of binary expression and values pair which corresponds to one
            case condition - value
        else_ (str | int | float | complex | bool): else value in case expression

    Returns:
        Func: A Func object that represents the case function.

    Example:
        ```py
        dc.mutate(
            res=func.case((C("num") > 0, "P"), (C("num") < 0, "N"), else_="Z"),
        )
        ```
    """
    supported_types = [int, float, complex, str, bool]

    type_ = type(else_) if else_ else None

    if not args:
        raise DataChainParamsError("Missing statements")

    for arg in args:
        if type_ and not isinstance(arg[1], type_):
            raise DataChainParamsError("Statement values must be of the same type")
        type_ = type(arg[1])

    if type_ not in supported_types:
        raise DataChainParamsError(
            f"Only python literals ({supported_types}) are supported for values"
        )

    kwargs = {"else_": else_}
    return Func("case", inner=sql_case, args=args, kwargs=kwargs, result_type=type_)


def ifelse(condition: BinaryExpression, if_val: CaseT, else_val: CaseT) -> Func:
    """
    Returns the ifelse function that produces if expression which has a condition
    and values for true and false outcome. Results can only be python primitives
    like string, numbes or booleans. Result type is inferred from the values.

    Args:
        condition: BinaryExpression - condition which is evaluated
        if_val: (str | int | float | complex | bool): value for true condition outcome
        else_val: (str | int | float | complex | bool): value for false condition
         outcome

    Returns:
        Func: A Func object that represents the ifelse function.

    Example:
        ```py
        dc.mutate(
            res=func.ifelse(C("num") > 0, "P", "N"),
        )
        ```
    """
    return case((condition, if_val), else_=else_val)


def isnone(col: Union[str, Column]) -> Func:
    """
    Returns True if column value or literal is None, otherwise False
    Args:
        col (str | Column | literal): Column or literal to check if None.
            If a string is provided, it is assumed to be the name of the column.
            If a literal is provided, it is assumed to be a string literal.

    Returns:
        Func: A Func object that represents the conditional to check if column is None.

    Example:
        ```py
        dc.mutate(test=isnone("value"))
        ```
    """
    from datachain import C

    if isinstance(col, str):
        # if string, it is assumed to be the name of the column
        col = C(col)

    return case((col == None, True), else_=False)  # type: ignore [arg-type]  # noqa: E711
