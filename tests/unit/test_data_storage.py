import math
from datetime import datetime, timezone
from typing import Any

import pytest
import sqlalchemy

from datachain.error import OutdatedDatabaseSchemaError
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
    UInt32,
    UInt64,
)
from tests.utils import (
    DEFAULT_TREE,
    TARRED_TREE,
    skip_if_not_sqlite,
)

COMPLEX_TREE: dict[str, Any] = {
    **TARRED_TREE,
    **DEFAULT_TREE,
    "nested": {"dir": {"path": {"abc.txt": "abc"}}},
}


@pytest.mark.parametrize(
    "col_type,default_value",
    [
        [String(), ""],
        [Boolean(), False],
        [Int(), 0],
        [Int32(), 0],
        [UInt32(), 0],
        [Int64(), 0],
        [UInt64(), 0],
        [Float(), lambda val: math.isnan(val)],
        [Float32(), lambda val: math.isnan(val)],
        [Float64(), lambda val: math.isnan(val)],
        [Array(Int), []],
        [JSON(), {}],
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


@skip_if_not_sqlite
def test_outdated_schema(catalog):
    from datachain.data_storage.sqlite import SCHEMA_VERSION

    metastore = catalog.metastore

    metastore._check_schema_version()  # should not raise exception

    # update schema version to be lower than current one
    stmt = (
        metastore._meta.update()
        .where(metastore._meta.c.id == 1)
        .values(schema_version=SCHEMA_VERSION - 1)
    )
    metastore.db.execute(stmt)

    with pytest.raises(OutdatedDatabaseSchemaError):
        metastore._check_schema_version()


@skip_if_not_sqlite
def test_outdated_schema_meta_not_present(catalog):
    metastore = catalog.metastore
    metastore.db.drop_table(metastore._meta)

    with pytest.raises(OutdatedDatabaseSchemaError):
        metastore._init_meta_table()
