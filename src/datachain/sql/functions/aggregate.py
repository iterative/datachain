from sqlalchemy.sql.functions import GenericFunction, ReturnTypeFromArgs

from datachain.sql.types import Float, String
from datachain.sql.utils import compiler_not_implemented


class avg(GenericFunction):  # noqa: N801
    """
    Returns the average of the column.
    """

    type = Float()
    package = "array"
    name = "avg"
    inherit_cache = True


class group_concat(GenericFunction):  # noqa: N801
    """
    Returns the concatenated string of the column.
    """

    type = String()
    package = "array"
    name = "group_concat"
    inherit_cache = True


class any_value(ReturnTypeFromArgs):  # noqa: N801
    """
    Returns first value of the column.
    """

    inherit_cache = True


class collect(ReturnTypeFromArgs):  # noqa: N801
    """
    Returns an array of the column.
    """

    inherit_cache = True


compiler_not_implemented(avg)
compiler_not_implemented(group_concat)
compiler_not_implemented(any_value)
