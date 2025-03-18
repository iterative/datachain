from sqlalchemy.sql.functions import GenericFunction, ReturnTypeFromArgs

from datachain.sql.types import Int64
from datachain.sql.utils import compiler_not_implemented


class bit_and(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_or(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_xor(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_rshift(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class bit_lshift(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class int_hash_64(GenericFunction):  # noqa: N801
    """
    Computes the 64-bit hash of an integer.
    """

    type = Int64()
    package = "hash"
    name = "int_hash_64"
    inherit_cache = True


class bit_hamming_distance(GenericFunction):  # noqa: N801
    """
    Returns the Hamming distance between two integers.
    """

    type = Int64()
    package = "numeric"
    name = "hamming_distance"
    inherit_cache = True


compiler_not_implemented(bit_and)
compiler_not_implemented(bit_or)
compiler_not_implemented(bit_xor)
compiler_not_implemented(bit_rshift)
compiler_not_implemented(bit_lshift)
compiler_not_implemented(int_hash_64)
compiler_not_implemented(bit_hamming_distance)
