from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.schema import CreateTable

from datachain.data_storage.schema import DataTable
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    Float,
    Float32,
    Float64,
    Int,
    Int64,
    String,
)


def test_dataset_table_compilation():
    table = DataTable.new_table(
        "ds-1",
        columns=[
            Column("vtype", String, nullable=False, index=True),
            Column("dir_type", Int, index=True),
            Column("path", String, nullable=False, index=True),
            Column("etag", String),
            Column("version", String),
            Column("is_latest", Boolean),
            Column("last_modified", DateTime(timezone=True)),
            Column("size", Int64, nullable=False, index=True),
            Column("owner_name", String),
            Column("owner_id", String),
            Column("location", JSON),
            Column("source", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("meta_info", JSON),
        ],
    )
    result = CreateTable(table, if_not_exists=True).compile(dialect=sqlite_dialect())

    assert result.string == (
        "\n"
        'CREATE TABLE IF NOT EXISTS "ds-1" (\n'
        "\tsys__id INTEGER NOT NULL, \n"
        "\tsys__rand INTEGER DEFAULT (abs(random())) NOT NULL, \n"
        "\tvtype VARCHAR NOT NULL, \n"
        "\tdir_type INTEGER, \n"
        "\tpath VARCHAR NOT NULL, \n"
        "\tetag VARCHAR, \n"
        "\tversion VARCHAR, \n"
        "\tis_latest BOOLEAN, \n"
        "\tlast_modified DATETIME, \n"
        "\tsize INTEGER NOT NULL, \n"
        "\towner_name VARCHAR, \n"
        "\towner_id VARCHAR, \n"
        "\tlocation JSON, \n"
        "\tsource VARCHAR NOT NULL, \n"
        "\tscore FLOAT NOT NULL, \n"
        "\tmeta_info JSON, \n"
        "\tPRIMARY KEY (sys__id)\n"
        ")\n"
        "\n"
    )


def test_schema_serialization(dataset_record):
    dataset_record.schema = {"int_col": Int}
    assert dataset_record.serialized_schema == {"int_col": {"type": "Int"}}

    dataset_record.schema = {
        "binary_col": Binary,
        "float_32_col": Float32,
    }
    assert dataset_record.serialized_schema == {
        "binary_col": {"type": "Binary"},
        "float_32_col": {"type": "Float32"},
    }

    dataset_record.schema = {"nested_col": Array(Array(Float64))}
    assert dataset_record.serialized_schema == {
        "nested_col": {
            "type": "Array",
            "item_type": {"type": "Array", "item_type": {"type": "Float64"}},
        }
    }
