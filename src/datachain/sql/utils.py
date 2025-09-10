import hashlib
import json
from collections.abc import Sequence
from typing import Union

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    ClauseElement,
    ColumnElement,
    Label,
    Over,
    UnaryExpression,
)
from sqlalchemy.sql.functions import Function


def compiler_not_implemented(func, *spec):
    package = getattr(func, "package", None)
    if package is None:
        func_identifier = func.name
    else:
        func_identifier = f"{func.package}.{func.name}"

    @compiles(func, *spec)
    def raise_not_implemented(element, compiler, **kwargs):
        try:
            dialect_name = compiler.dialect.name
        except AttributeError:
            dialect_name = "unknown"
        raise NotImplementedError(
            f"Compiler not implemented for the SQLAlchemy function, {func_identifier},"
            f" with dialect, {dialect_name}. For information on adding dialect-specific"
            " compilers, see https://docs.sqlalchemy.org/en/14/core/compiler.html"
        )

    return raise_not_implemented


def serialize_expression(expr: Union[str, ClauseElement]) -> dict:  # noqa: PLR0911
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
            "left": serialize_expression(expr.left),
            "right": serialize_expression(expr.right),
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
            "element": serialize_expression(expr.element),
        }

    # Function calls: func.lower(col), func.count(col), etc.
    if isinstance(expr, Function):
        return {
            "type": "function",
            "name": expr.name,
            "clauses": [serialize_expression(c) for c in expr.clauses],
        }

    # Window functions: func.row_number().over(partition_by=..., order_by=...)
    if isinstance(expr, Over):
        return {
            "type": "window",
            "function": serialize_expression(expr.element),
            "partition_by": [
                serialize_expression(p) for p in getattr(expr, "partition_by", [])
            ],
            "order_by": [
                serialize_expression(o) for o in getattr(expr, "order_by", [])
            ],
        }

    # Labeled expressions: col.label("alias")
    if isinstance(expr, Label):
        return {
            "type": "label",
            "name": expr.name,
            "element": serialize_expression(expr.element),
        }

    # Bound values (constants)
    if isinstance(expr, BindParameter):
        return {"type": "bind", "value": expr.value}

    # Plain columns
    if hasattr(expr, "name"):
        return {"type": "column", "name": expr.name}

    # Fallback: stringify unknown nodes
    return {"type": "other", "repr": str(expr)}


def hash_column_elements(columns: Sequence[Union[ColumnElement, str]]) -> str:
    """
    Hash a list of ColumnElements deterministically, dialect agnostic.
    Only accepts ordered iterables (like list or tuple).
    """
    serialized = [serialize_expression(c) for c in columns]
    json_str = json.dumps(serialized, sort_keys=True)  # stable JSON
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()
