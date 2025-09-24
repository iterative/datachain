import hashlib
import inspect
import json
import textwrap
from collections.abc import Sequence
from typing import TypeVar, Union

from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    ColumnElement,
    Label,
    Over,
    UnaryExpression,
)
from sqlalchemy.sql.functions import Function

T = TypeVar("T", bound=ColumnElement)
ColumnLike = Union[str, T]


def serialize_column_element(expr: Union[str, ColumnElement]) -> dict:  # noqa: PLR0911
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

    # Function calls: func.lower(col), func.count(col), etc.
    if isinstance(expr, Function):
        return {
            "type": "function",
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


def hash_column_elements(columns: Sequence[ColumnLike]) -> str:
    """
    Hash a list of ColumnElements deterministically, dialect agnostic.
    Only accepts ordered iterables (like list or tuple).
    """
    serialized = [serialize_column_element(c) for c in columns]
    json_str = json.dumps(serialized, sort_keys=True)  # stable JSON
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def hash_callable(func):
    """
    Calculate a hash from a callable.
    Rules:
    - Named functions (def) → use source code for stable, cross-version hashing
    - Lambdas → use bytecode (deterministic in same Python runtime)
    """
    if not callable(func):
        raise TypeError("Expected a callable")

    # Determine if it is a lambda
    is_lambda = func.__name__ == "<lambda>"

    if not is_lambda:
        # Try to get exact source of named function
        try:
            lines, _ = inspect.getsourcelines(func)
            payload = textwrap.dedent("".join(lines)).strip()
        except (OSError, TypeError):
            # Fallback: bytecode if source not available
            payload = func.__code__.co_code
    else:
        # For lambdas, fall back directly to bytecode
        payload = func.__code__.co_code

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

    # Compute SHA256
    h = hashlib.sha256()
    h.update(str(payload).encode() if isinstance(payload, str) else payload)
    h.update(str(extras).encode())
    return h.hexdigest()
