from sqlalchemy.sql.functions import GenericFunction

from datachain.sql.types import Array, Int64, String
from datachain.sql.utils import compiler_not_implemented


class length(GenericFunction):  # noqa: N801
    type = Int64()
    package = "string"
    name = "length"
    inherit_cache = True


class split(GenericFunction):  # noqa: N801
    type = Array(String())
    package = "string"
    name = "split"
    inherit_cache = True


compiler_not_implemented(length)
compiler_not_implemented(split)
