from sqlalchemy.sql.sqltypes import NullType

from datachain import Column
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.sql import functions as func
from datachain.sql.types import Float, Int64, String


def test_sql_columns_to_python_types():
    assert sql_to_python(
        {
            "name": Column("name", String),
            "age": Column("age", Int64),
            "score": Column("score", Float),
        }
    ) == {"name": str, "age": int, "score": float}


def test_sql_expression_to_python_types():
    assert sql_to_python({"age": Column("age", Int64) - 2}) == {"age": int}


def test_sql_function_to_python_types():
    assert sql_to_python({"age": func.avg(Column("age", Int64))}) == {"age": float}


def test_sql_to_python_types_default_type():
    assert sql_to_python({"null": Column("null", NullType)}) == {"null": str}
