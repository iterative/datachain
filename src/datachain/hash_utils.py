import hashlib
import inspect
import json
import textwrap
from collections.abc import Sequence
from typing import TypeVar

from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    ColumnElement,
    Label,
    Over,
    UnaryExpression,
)
from sqlalchemy.sql.functions import Function as SAFunction

from datachain.func.base import Function

T = TypeVar("T", bound=ColumnElement)


def serialize_column_element(expr: str | Function | ColumnElement) -> dict:  # noqa: PLR0911
    """
    Recursively serialize a SQLAlchemy ColumnElement into a deterministic structure.
    """

    # Binary operations: col > 5, col1 + col2, etc.
    if isinstance(expr, BinaryExpression):
        op = (
            expr.operator.__name__
            if hasattr(expr.operator, "__name__")
            else str(expr.operator)
        )
        return {
            "type": "binary",
            "op": op,
            "left": serialize_column_element(expr.left),
            "right": serialize_column_element(expr.right),
        }

    # Unary operations: -col, NOT col, etc.
    if isinstance(expr, UnaryExpression):
        op = (
            expr.operator.__name__
            if expr.operator is not None and hasattr(expr.operator, "__name__")
            else str(expr.operator)
        )

        return {
            "type": "unary",
            "op": op,
            "element": serialize_column_element(expr.element),  # type: ignore[arg-type]
        }

    # Datachain functions: Function("my_func", col1, col2)
    if isinstance(expr, Function):
        return {
            "type": "dc-function",
            "name": expr.name,
            "clauses": [serialize_column_element(c) for c in [*expr.cols, *expr.args]],
        }

    # Function calls: func.lower(col), func.count(col), etc.
    if isinstance(expr, SAFunction):
        return {
            "type": "sa-function",
            "name": expr.name,
            "clauses": [serialize_column_element(c) for c in expr.clauses],
        }

    # Window functions: func.row_number().over(partition_by=..., order_by=...)
    if isinstance(expr, Over):
        return {
            "type": "window",
            "function": serialize_column_element(expr.element),
            "partition_by": [
                serialize_column_element(p) for p in getattr(expr, "partition_by", [])
            ],
            "order_by": [
                serialize_column_element(o) for o in getattr(expr, "order_by", [])
            ],
        }

    # Labeled expressions: col.label("alias")
    if isinstance(expr, Label):
        return {
            "type": "label",
            "name": expr.name,
            "element": serialize_column_element(expr.element),
        }

    # Bound values (constants)
    if isinstance(expr, BindParameter):
        return {"type": "bind", "value": expr.value}

    # Plain columns
    if hasattr(expr, "name"):
        return {"type": "column", "name": expr.name}

    # Fallback: stringify unknown nodes
    return {"type": "other", "repr": str(expr)}


def hash_column_elements(columns: Sequence[str | Function | T]) -> str:
    """
    Hash a list of ColumnElements deterministically, dialect agnostic.
    Only accepts ordered iterables (like list or tuple).
    """
    serialized = [serialize_column_element(c) for c in columns]
    json_str = json.dumps(serialized, sort_keys=True)  # stable JSON
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def hash_callable(func, _visited=None):
    """
    Calculate a hash from a callable, including its dependencies.
    Rules:
    - Named functions (def) → use source code for stable, cross-version hashing
    - Lambdas → use bytecode (deterministic in same Python runtime)
    - Recursively hashes helper functions from the same module
    """
    if not callable(func):
        raise TypeError("Expected a callable")

    # Track visited functions to avoid infinite recursion
    if _visited is None:
        _visited = set()

    # Use id(func) to track which functions we've visited
    func_id = id(func)
    if func_id in _visited:
        return hashlib.sha256(f"recursive:{func.__name__}".encode()).hexdigest()
    _visited.add(func_id)

    # Determine if it is a lambda
    is_lambda = func.__name__ == "<lambda>"

    if not is_lambda:
        # Try to get exact source of named function
        try:
            lines, _ = inspect.getsourcelines(func)
            payload = textwrap.dedent("".join(lines)).strip()
        except (OSError, TypeError):
            # Fallback: bytecode + constants if source not available
            code = func.__code__
            payload = (code.co_code, code.co_consts, code.co_names, code.co_varnames)
    else:
        # For lambdas, use bytecode + constants
        code = func.__code__
        payload = (code.co_code, code.co_consts, code.co_names, code.co_varnames)

    # Normalize annotations
    annotations = {
        k: getattr(v, "__name__", str(v)) for k, v in func.__annotations__.items()
    }

    # Extras to distinguish functions with same code but different metadata
    extras = {
        "name": func.__name__,
        "defaults": func.__defaults__,
        "annotations": annotations,
    }

    # Find helper functions that this function depends on
    dependencies = {}
    if hasattr(func, "__code__") and hasattr(func, "__globals__"):
        # Get all names referenced in the function's code
        referenced_names = func.__code__.co_names
        func_module = inspect.getmodule(func)

        for name in referenced_names:
            # Look up the name in the function's global namespace
            if name in func.__globals__:
                obj = func.__globals__[name]

                # Only hash user-defined functions from the same module
                # Skip built-ins, imported functions from other modules, and classes
                if (
                    callable(obj)
                    and hasattr(obj, "__module__")
                    and func_module is not None
                    and obj.__module__ == func_module.__name__
                    and not inspect.isclass(obj)
                    and not inspect.isbuiltin(obj)
                ):
                    # Recursively hash the dependency
                    try:
                        dependencies[name] = hash_callable(obj, _visited)
                    except (TypeError, OSError):
                        # If we can't hash it, skip it
                        pass

    # Compute SHA256
    h = hashlib.sha256()
    if isinstance(payload, str):
        h.update(payload.encode())
    else:
        # payload is a tuple of (bytecode, consts, names, varnames)
        h.update(str(payload).encode())
    h.update(str(extras).encode())
    # Include dependency hashes in sorted order for determinism
    if dependencies:
        deps_str = json.dumps(dependencies, sort_keys=True)
        h.update(deps_str.encode())
    return h.hexdigest()
