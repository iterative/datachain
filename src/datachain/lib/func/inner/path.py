"""
This module provides generic SQL functions for path logic.

These need to be implemented using dialect-specific compilation rules.
See https://docs.sqlalchemy.org/en/14/core/compiler.html
"""

from sqlalchemy.sql.functions import GenericFunction

from datachain.sql.types import String
from datachain.sql.utils import compiler_not_implemented


class parent(GenericFunction):  # noqa: N801
    """
    Returns the directory component of a posix-style path.
    """

    type = String()
    package = "path"
    name = "parent"
    inherit_cache = True


class name(GenericFunction):  # noqa: N801
    """
    Returns the final component of a posix-style path.
    """

    type = String()
    package = "path"
    name = "name"
    inherit_cache = True


class file_stem(GenericFunction):  # noqa: N801
    """
    Strips an extension from the given path.
    """

    type = String()
    package = "path"
    name = "file_stem"
    inherit_cache = True


class file_ext(GenericFunction):  # noqa: N801
    """
    Returns the extension of the given path.
    """

    type = String()
    package = "path"
    name = "file_ext"
    inherit_cache = True


compiler_not_implemented(parent)
compiler_not_implemented(name)
compiler_not_implemented(file_stem)
compiler_not_implemented(file_ext)
