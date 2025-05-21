from sqlalchemy.sql.functions import GenericFunction

from datachain.sql.types import Boolean, Float, Int64, String
from datachain.sql.utils import compiler_not_implemented


class cosine_distance(GenericFunction):  # noqa: N801
    """
    Takes a column and array and returns the cosine distance between them.
    """

    type = Float()
    package = "array"
    name = "cosine_distance"
    inherit_cache = True


class euclidean_distance(GenericFunction):  # noqa: N801
    """
    Takes a column and array and returns the Euclidean distance between them.
    """

    type = Float()
    package = "array"
    name = "euclidean_distance"
    inherit_cache = True


class length(GenericFunction):  # noqa: N801
    """
    Returns the length of the array.
    """

    type = Int64()
    package = "array"
    name = "length"
    inherit_cache = True


class contains(GenericFunction):  # noqa: N801
    """
    Checks if element is in the array.
    """

    type = Boolean()
    package = "array"
    name = "contains"
    inherit_cache = True


class slice(GenericFunction):  # noqa: N801
    """
    Returns a slice of the array.
    """

    package = "array"
    name = "slice"
    inherit_cache = True


class join(GenericFunction):  # noqa: N801
    """
    Returns the concatenation of the array elements.
    """

    type = String()
    package = "array"
    name = "join"
    inherit_cache = True


class get_element(GenericFunction):  # noqa: N801
    """
    Returns the element at the given index in the array.
    """

    package = "array"
    name = "get_element"
    inherit_cache = True


class sip_hash_64(GenericFunction):  # noqa: N801
    """
    Computes the SipHash-64 hash of the array.
    """

    type = Int64()
    package = "hash"
    name = "sip_hash_64"
    inherit_cache = True


compiler_not_implemented(cosine_distance)
compiler_not_implemented(euclidean_distance)
compiler_not_implemented(length)
compiler_not_implemented(contains)
compiler_not_implemented(get_element)
compiler_not_implemented(sip_hash_64)
