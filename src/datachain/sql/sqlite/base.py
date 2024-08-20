import logging
import re
import sqlite3
from collections.abc import Iterable
from datetime import MAXYEAR, MINYEAR, datetime, timezone
from types import MappingProxyType
from typing import Callable, Optional

import orjson
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import literal
from sqlalchemy.sql.expression import case
from sqlalchemy.sql.functions import func

from datachain.sql.functions import array, conditional, random, string
from datachain.sql.functions import path as sql_path
from datachain.sql.selectable import Values, base_values_compiler
from datachain.sql.sqlite.types import (
    SQLiteTypeConverter,
    SQLiteTypeReadConverter,
    register_type_converters,
)
from datachain.sql.types import (
    DBDefaults,
    TypeDefaults,
    register_backend_types,
    register_db_defaults,
    register_type_defaults,
    register_type_read_converters,
)

logger = logging.getLogger("datachain")

_registered_function_creators: dict[str, Callable[[sqlite3.Connection], None]] = {}
registered_function_creators = MappingProxyType(_registered_function_creators)

_compiler_hooks: dict[str, Callable] = {}

sqlite_dialect = sqlite.dialect(paramstyle="named")

setup_is_complete: bool = False

slash = literal("/")
empty_str = literal("")
dot = literal(".")


def setup():
    global setup_is_complete  # noqa: PLW0603
    if setup_is_complete:
        return

    # sqlite 3.31.1 is the earliest version tested in CI
    if sqlite3.sqlite_version_info < (3, 31, 1):
        logger.warning(
            "Possible sqlite incompatibility. The earliest tested version of "
            "sqlite is 3.31.1 but you have %s",
            sqlite3.sqlite_version,
        )

    # We want to show tracebacks for user-defined functions
    sqlite3.enable_callback_tracebacks(True)
    sqlite3.register_adapter(datetime, adapt_datetime)
    sqlite3.register_converter("datetime", convert_datetime)

    register_type_converters()
    register_backend_types("sqlite", SQLiteTypeConverter())
    register_type_read_converters("sqlite", SQLiteTypeReadConverter())
    register_type_defaults("sqlite", TypeDefaults())
    register_db_defaults("sqlite", DBDefaults())

    compiles(sql_path.parent, "sqlite")(compile_path_parent)
    compiles(sql_path.name, "sqlite")(compile_path_name)
    compiles(sql_path.file_stem, "sqlite")(compile_path_file_stem)
    compiles(sql_path.file_ext, "sqlite")(compile_path_file_ext)
    compiles(array.length, "sqlite")(compile_array_length)
    compiles(string.length, "sqlite")(compile_string_length)
    compiles(string.split, "sqlite")(compile_string_split)
    compiles(string.regexp_replace, "sqlite")(compile_regexp_replace)
    compiles(conditional.greatest, "sqlite")(compile_greatest)
    compiles(conditional.least, "sqlite")(compile_least)
    compiles(Values, "sqlite")(compile_values)
    compiles(random.rand, "sqlite")(compile_rand)
    compiles(array.avg, "sqlite")(compile_avg)

    if load_usearch_extension(sqlite3.connect(":memory:")):
        compiles(array.cosine_distance, "sqlite")(compile_cosine_distance_ext)
        compiles(array.euclidean_distance, "sqlite")(compile_euclidean_distance_ext)
    else:
        compiles(array.cosine_distance, "sqlite")(compile_cosine_distance)
        compiles(array.euclidean_distance, "sqlite")(compile_euclidean_distance)

    register_user_defined_sql_functions()
    setup_is_complete = True


def run_compiler_hook(name):
    try:
        hook = _compiler_hooks[name]
    except KeyError:
        return
    hook()


def functions_exist(
    names: Iterable[str], connection: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Returns True if all function names are defined for the given connection.
    """

    names = list(names)
    for n in names:
        if not isinstance(n, str):
            raise TypeError(
                "functions_exist(): names argument must contain str values. "
                f"Found value of type {type(n).__name__}: {n!r}"
            )

    if connection is None:
        connection = sqlite3.connect(":memory:")

    if not names:
        return True
    column1 = sa.column("column1", sa.String)
    func_name_query = column1.not_in(
        sa.select(sa.column("name", sa.String)).select_from(func.pragma_function_list())
    )
    query = (
        sa.select(func.count() == 0)
        .select_from(sa.values(column1).data([(n,) for n in names]))
        .where(func_name_query)
    )
    comp = query.compile(dialect=sqlite_dialect)
    args = (comp.string, comp.params) if comp.params else (comp.string,)
    return bool(connection.execute(*args).fetchone()[0])


def create_user_defined_sql_functions(connection):
    for function_creator in registered_function_creators.values():
        function_creator(connection)


def missing_vector_function(name, exc):
    def unavailable_func(*args):
        raise ImportError(
            f"Missing dependencies for SQL vector function, {name}\n"
            "To install run:\n\n"
            "  pip install 'datachain[vector]'\n"
        ) from exc

    return unavailable_func


def sqlite_string_split(string: str, sep: str, maxsplit: int = -1) -> str:
    return orjson.dumps(string.split(sep, maxsplit)).decode("utf-8")


def register_user_defined_sql_functions() -> None:
    # Register optional functions if we have the necessary dependencies
    # and otherwise register functions that will raise an exception with
    # installation instructions
    try:
        from .vector import cosine_distance, euclidean_distance
    except ImportError as exc:
        # We want to throw an exception when trying to compile these
        # functions and also if the functions are called using raw SQL.
        cosine_distance = missing_vector_function("cosine_distance", exc)
        euclidean_distance = missing_vector_function("euclidean_distance", exc)
        _compiler_hooks["cosine_distance"] = cosine_distance
        _compiler_hooks["euclidean_distance"] = euclidean_distance

    def create_vector_functions(conn):
        conn.create_function("cosine_distance", 2, cosine_distance, deterministic=True)
        conn.create_function(
            "euclidean_distance", 2, euclidean_distance, deterministic=True
        )

    _registered_function_creators["vector_functions"] = create_vector_functions

    def sqlite_regexp_replace(string: str, pattern: str, replacement: str) -> str:
        return re.sub(pattern, replacement, string)

    def create_string_functions(conn):
        conn.create_function("split", 2, sqlite_string_split, deterministic=True)
        conn.create_function("split", 3, sqlite_string_split, deterministic=True)
        conn.create_function(
            "regexp_replace", 3, sqlite_regexp_replace, deterministic=True
        )

    _registered_function_creators["string_functions"] = create_string_functions

    has_json_extension = functions_exist(["json_array_length"])
    if not has_json_extension:

        def create_json_functions(conn):
            conn.create_function(
                "json_array_length", 1, py_json_array_length, deterministic=True
            )

        _registered_function_creators["json_functions"] = create_json_functions


def adapt_datetime(val: datetime) -> str:
    if not (val.tzinfo is timezone.utc or val.tzname() == "UTC"):
        try:
            val = val.astimezone(timezone.utc)
        except (OverflowError, ValueError, OSError):
            if val.year == MAXYEAR:
                val = datetime.max
            elif val.year == MINYEAR:
                val = datetime.min
            else:
                raise
    return val.replace(tzinfo=None).isoformat(" ")


def convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode()).replace(tzinfo=timezone.utc)


def path_parent(path):
    return func.rtrim(func.rtrim(path, func.replace(path, slash, empty_str)), slash)


def path_name(path):
    return func.ltrim(func.substr(path, func.length(path_parent(path)) + 1), slash)


def name_file_ext_length(name):
    expr = func.length(name) - func.length(
        func.rtrim(name, func.replace(name, dot, empty_str))
    )
    return case((func.instr(name, dot) == 0, 0), else_=expr)


def path_file_ext_length(path):
    name = path_name(path)
    return name_file_ext_length(name)


def path_file_stem(path):
    path_length = func.length(path)
    parent_length = func.length(path_parent(path))

    name_expr = func.rtrim(
        func.substr(
            path,
            1,
            path_length - name_file_ext_length(path),
        ),
        dot,
    )

    full_path_expr = func.ltrim(
        func.rtrim(
            func.substr(
                path,
                parent_length + 1,
                path_length - parent_length - path_file_ext_length(path),
            ),
            dot,
        ),
        slash,
    )

    return case((func.instr(path, slash) == 0, name_expr), else_=full_path_expr)


def path_file_ext(path):
    return func.substr(path, func.length(path) - path_file_ext_length(path) + 1)


def compile_regexp_replace(element, compiler, **kwargs):
    return f"regexp_replace({compiler.process(element.clauses, **kwargs)})"


def compile_path_parent(element, compiler, **kwargs):
    return compiler.process(path_parent(*element.clauses.clauses), **kwargs)


def compile_path_name(element, compiler, **kwargs):
    return compiler.process(path_name(*element.clauses.clauses), **kwargs)


def compile_path_file_stem(element, compiler, **kwargs):
    return compiler.process(path_file_stem(*element.clauses.clauses), **kwargs)


def compile_path_file_ext(element, compiler, **kwargs):
    return compiler.process(path_file_ext(*element.clauses.clauses), **kwargs)


def compile_cosine_distance_ext(element, compiler, **kwargs):
    run_compiler_hook("cosine_distance")
    return f"distance_cosine_f32({compiler.process(element.clauses, **kwargs)})"


def compile_cosine_distance(element, compiler, **kwargs):
    run_compiler_hook("cosine_distance")
    return f"cosine_distance({compiler.process(element.clauses, **kwargs)})"


def compile_euclidean_distance_ext(element, compiler, **kwargs):
    run_compiler_hook("euclidean_distance")
    return (
        f"sqrt(distance_sqeuclidean_f32({compiler.process(element.clauses, **kwargs)}))"
    )


def compile_euclidean_distance(element, compiler, **kwargs):
    run_compiler_hook("euclidean_distance")
    return f"euclidean_distance({compiler.process(element.clauses, **kwargs)})"


def py_json_array_length(arr):
    return len(orjson.loads(arr))


def compile_array_length(element, compiler, **kwargs):
    return compiler.process(func.json_array_length(*element.clauses.clauses), **kwargs)


def compile_string_length(element, compiler, **kwargs):
    return compiler.process(func.length(*element.clauses.clauses), **kwargs)


def compile_string_split(element, compiler, **kwargs):
    return compiler.process(func.split(*element.clauses.clauses), **kwargs)


def compile_greatest(element, compiler, **kwargs):
    """
    Compiles a sql function for `greatest(*args)` taking 1 or more args

    Compiles to:
      - `max(arg1, arg2...)` for 2 or more args
      - `arg1` for 1 arg

    sqlite's max() is a simple function when it has 2 or more
    arguments but operates as an aggregate function if given only a
    single argument
    See https://www.sqlite.org/lang_corefunc.html#max_scalar
    """
    args = element.clauses.clauses
    nargs = len(args)
    if nargs < 1:
        raise TypeError(
            f"conditional.greatest requires at least 1 argument ({nargs} found)"
        )
    if nargs == 1:
        expr = args[0]
    else:
        expr = func.max(*args)
    return compiler.process(expr, **kwargs)


def compile_least(element, compiler, **kwargs):
    """
    Compiles a sql function for `least(*args)` taking 1 or more args

    Compiles to:
      - `min(arg1, arg2...)` for 2 or more args
      - `arg1` for 1 arg

    sqlite's min() is a simple function when it has 2 or more
    arguments but operates as an aggregate function if given only a
    single argument
    See https://www.sqlite.org/lang_corefunc.html#min_scalar
    """
    args = element.clauses.clauses
    nargs = len(args)
    if nargs < 1:
        raise TypeError(
            f"conditional.least requires at least 1 argument ({nargs} found)"
        )
    if nargs == 1:
        expr = args[0]
    else:
        expr = func.min(*args)
    return compiler.process(expr, **kwargs)


def compile_values(element, compiler, **kwargs):
    return base_values_compiler(lambda i: f"column{i}", element, compiler, **kwargs)


def compile_rand(element, compiler, **kwargs):
    return compiler.process(func.random(), **kwargs)


def compile_avg(element, compiler, **kwargs):
    return compiler.process(func.avg(*element.clauses.clauses), **kwargs)


def load_usearch_extension(conn) -> bool:
    try:
        # usearch is part of the vector optional dependencies
        # we use the extension's cosine and euclidean distance functions
        from usearch import sqlite_path

        conn.enable_load_extension(True)
        conn.load_extension(sqlite_path())
        conn.enable_load_extension(False)
        return True

    except Exception:  # noqa: BLE001
        return False
