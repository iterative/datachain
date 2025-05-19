import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect
from sqlalchemy.schema import CreateTable

from datachain.data_storage.schema import DataTable
from datachain.dataset import DatasetDependency, DatasetDependencyType, DatasetVersion
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
            Column("dir_type", Int, index=True),
            Column("path", String, nullable=False, index=True),
            Column("etag", String),
            Column("version", String),
            Column("is_latest", Boolean),
            Column("last_modified", DateTime(timezone=True)),
            Column("size", Int64, nullable=False, index=True),
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
        "\tdir_type INTEGER, \n"
        "\tpath VARCHAR NOT NULL, \n"
        "\tetag VARCHAR, \n"
        "\tversion VARCHAR, \n"
        "\tis_latest BOOLEAN, \n"
        "\tlast_modified DATETIME, \n"
        "\tsize INTEGER NOT NULL, \n"
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


@pytest.mark.parametrize(
    "dep_name,dep_type,expected",
    [
        ("dogs_dataset", DatasetDependencyType.DATASET, "dogs_dataset"),
        (
            "s3://dogs_dataset/dogs",
            DatasetDependencyType.STORAGE,
            "lst__s3://dogs_dataset/dogs/",
        ),
    ],
)
def test_dataset_dependency_dataset_name(dep_name, dep_type, expected):
    dep = DatasetDependency(
        id=1,
        name=dep_name,
        version="1.0.0",
        type=dep_type,
        created_at=datetime.now(timezone.utc),
        dependencies=[],
    )

    assert dep.dataset_name == expected


@pytest.mark.parametrize(
    "use_string",
    [True, False],
)
def test_dataset_version_from_dict(use_string):
    preview = [{"id": 1, "thing": "a"}, {"id": 2, "thing": "b"}]

    preview_data = json.dumps(preview) if use_string else preview

    data = {
        "id": 1,
        "uuid": "98928be4-b6e8-4b7b-a7c5-2ce3b33130d8",
        "dataset_id": 40,
        "version": "2.0.0",
        "status": 1,
        "feature_schema": {},
        "created_at": datetime.fromisoformat("2023-10-01T12:00:00"),
        "finished_at": None,
        "error_message": "",
        "error_stack": "",
        "script_output": "",
        "schema": {},
        "num_objects": 100,
        "size": 1000000,
        "preview": preview_data,
    }

    dataset_version = DatasetVersion.from_dict(data)
    assert dataset_version.preview == preview
