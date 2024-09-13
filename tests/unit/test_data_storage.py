import math
from datetime import datetime, timezone
from typing import Any

import pytest
import sqlalchemy

from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    String,
    UInt64,
)
from tests.utils import DEFAULT_TREE, TARRED_TREE

COMPLEX_TREE: dict[str, Any] = {
    **TARRED_TREE,
    **DEFAULT_TREE,
    "nested": {"dir": {"path": {"abc.txt": "abc"}}},
}


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_convert_type(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    warehouse = catalog.warehouse
    now = datetime.now()

    def run_convert_type(value, sql_type):
        return warehouse.convert_type(
            value,
            sql_type,
            warehouse.python_type(sql_type),
            type(sql_type).__name__,
            "test_column",
        )

    # convert int to float
    for f in [Float, Float32, Float64]:
        converted = run_convert_type(1, f())
        assert converted == 1.0
        assert isinstance(converted, float)

    # types match, nothing to convert
    assert run_convert_type(1, Int()) == 1
    assert run_convert_type(1.5, Float()) == 1.5
    assert run_convert_type(True, Boolean()) is True
    assert run_convert_type("s", String()) == "s"
    assert run_convert_type(now, DateTime()) == now
    assert run_convert_type([1, 2], Array(Int)) == [1, 2]
    assert run_convert_type([1.5, 2.5], Array(Float)) == [1.5, 2.5]
    assert run_convert_type(["a", "b"], Array(String)) == ["a", "b"]
    assert run_convert_type([[1, 2], [3, 4]], Array(Array(Int))) == [
        [1, 2],
        [3, 4],
    ]

    # JSON Tests
    assert run_convert_type('{"a": 1}', JSON()) == '{"a": 1}'
    assert run_convert_type({"a": 1}, JSON()) == '{"a": 1}'
    assert run_convert_type([{"a": 1}], JSON()) == '[{"a": 1}]'
    with pytest.raises(ValueError):
        run_convert_type(0.5, JSON())

    # convert array to compatible type
    converted = run_convert_type([1, 2], Array(Float))
    assert converted == [1.0, 2.0]
    assert all(isinstance(c, float) for c in converted)

    # convert nested array to compatible type
    converted = run_convert_type([[1, 2], [3, 4]], Array(Array(Float)))
    assert converted == [[1.0, 2.0], [3.0, 4.0]]
    assert all(isinstance(c, float) for c in converted[0])
    assert all(isinstance(c, float) for c in converted[1])

    # error, float to int
    with pytest.raises(ValueError):
        run_convert_type(1.5, Int())

    # error, float to int in list
    with pytest.raises(ValueError):
        run_convert_type([1.5, 1], Array(Int))


@pytest.mark.parametrize(
    "col_type,default_value",
    [
        [String(), ""],
        [Boolean(), False],
        [Int(), 0],
        [Int32(), 0],
        [Int64(), 0],
        [UInt64(), 0],
        [Float(), lambda val: math.isnan(val)],
        [Float32(), lambda val: math.isnan(val)],
        [Float64(), lambda val: math.isnan(val)],
        [Array(Int), []],
        [JSON(), "{}"],
        [DateTime(), datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)],
        [Binary(), b""],
    ],
)
def test_db_defaults(col_type, default_value, catalog):
    warehouse = catalog.warehouse

    table_col = sqlalchemy.Column(
        "val",
        col_type,
        nullable=False,
        server_default=col_type.db_default_value(warehouse.db.dialect),
    )
    table = warehouse.create_udf_table([table_col])
    warehouse.insert_rows(table, [{"sys__id": 1}])
    warehouse.insert_rows_done(table)

    query = sqlalchemy.Select(table_col).select_from(table)

    values = [row[0] for row in warehouse.dataset_rows_select(query)]
    assert len(values) == 1
    if callable(default_value):
        assert default_value(values[0])
    else:
        assert values[0] == default_value

    warehouse.db.drop_table(table)
