from sqlalchemy.sql.functions import ReturnTypeFromArgs

from datachain.sql.utils import compiler_not_implemented


class bit_and(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_or(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_xor(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


compiler_not_implemented(bit_and)
compiler_not_implemented(bit_or)
compiler_not_implemented(bit_xor)
