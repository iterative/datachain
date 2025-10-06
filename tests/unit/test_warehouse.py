import base64
import json
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from datachain.data_storage.serializer import deserialize
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib import dc


@pytest.fixture
def numbers_ds(test_session):
    ds = dc.read_values(num=range(10), session=test_session).save("numbers_dataset")
    assert ds.dataset is not None
    yield ds.dataset
    dc.delete_dataset(ds.dataset.name, force=True)


@pytest.fixture
def numbers_table(warehouse, numbers_ds):
    table_name = warehouse.dataset_table_name(numbers_ds, numbers_ds.latest_version)
    table = warehouse.get_table(table_name)
    yield table


def test_serialize(sqlite_db):
    obj = SQLiteWarehouse(sqlite_db)
    assert obj.db == sqlite_db

    # Test clone
    obj2 = obj.clone()
    assert isinstance(obj2, SQLiteWarehouse)
    assert obj2.db.db_file == sqlite_db.db_file
    assert obj2.clone_params() == obj.clone_params()

    # Test serialization JSON format
    serialized = obj.serialize()
    assert serialized
    raw = base64.b64decode(serialized.encode())
    data = json.loads(raw.decode())
    assert data["callable"] == "sqlite.warehouse.init_after_clone"
    assert data["args"] == []
    nested = data["kwargs"]["db_clone_params"]
    assert nested["callable"] == "sqlite.from_db_file"
    assert nested["args"] == [":memory:"]
    assert nested["kwargs"] == {}

    obj3 = deserialize(serialized)
    assert isinstance(obj3, SQLiteWarehouse)
    assert obj3.db.db_file == sqlite_db.db_file
    assert obj3.clone_params() == obj.clone_params()


def test_is_temp_table_name(warehouse):
    assert warehouse.is_temp_table_name("tmp_vc12F") is True
    assert warehouse.is_temp_table_name("udf_jh653") is True
    assert warehouse.is_temp_table_name("ds_my_dataset") is False
    assert warehouse.is_temp_table_name("src_my_bucket") is False
    assert warehouse.is_temp_table_name("ds_ds_my_query_script_1_1") is False


def test_query_count(numbers_table, warehouse):
    query = sa.select(numbers_table.c.num).where(numbers_table.c.num < 5)
    assert warehouse.query_count(query) == 5


def test_table_rows_count(numbers_table, warehouse):
    assert warehouse.table_rows_count(numbers_table) == 10


def test_dataset_select_paginated(numbers_table, warehouse):
    query = sa.select(numbers_table.c.sys__id, numbers_table.c.num).order_by(
        numbers_table.c.num
    )
    with patch.object(
        type(warehouse),
        attribute="dataset_rows_select",
        wraps=warehouse.dataset_rows_select,
    ) as mock_dataset_rows_select:
        rows = list(warehouse.dataset_select_paginated(query, page_size=3))
        assert mock_dataset_rows_select.call_count == 4  # 4 pages: 3 + 3 + 3 + 1
    assert len(rows) == 10
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == 10
    assert list(nums) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("limit", [7, 8, 9])
def test_dataset_select_paginated_with_limit(limit, numbers_table, warehouse):
    query = (
        sa.select(numbers_table.c.sys__id, numbers_table.c.num)
        .order_by(numbers_table.c.num)
        .limit(limit)
    )
    with patch.object(
        type(warehouse),
        attribute="dataset_rows_select",
        wraps=warehouse.dataset_rows_select,
    ) as mock_dataset_rows_select:
        rows = list(warehouse.dataset_select_paginated(query, page_size=3))
        assert mock_dataset_rows_select.call_count == 3  # 3 pages: 3 + 3 + [1 | 2 | 3]
    assert len(rows) == limit
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == limit
    assert list(nums) == list(range(limit))


def test_dataset_rows_select(numbers_table, warehouse):
    query = (
        sa.select(numbers_table.c.sys__id, numbers_table.c.num)
        .order_by(numbers_table.c.num)
        .limit(7)
    )
    rows = list(warehouse.dataset_rows_select(query))
    assert len(rows) == 7
    ids, nums = zip(*rows, strict=False)
    assert len(set(ids)) == 7
    assert list(nums) == [0, 1, 2, 3, 4, 5, 6]


def test_dataset_rows_select_from_ids(numbers_table, warehouse):
    query_ids = (
        sa.select(numbers_table.c.sys__id).order_by(numbers_table.c.num).limit(5)
    )
    test_ids = [r[0] for r in warehouse.db.execute(query_ids)]
    assert len(test_ids) == len(set(test_ids)) == 5

    query = sa.select(numbers_table.c.sys__id, numbers_table.c.num)
    rows = list(
        warehouse.dataset_rows_select_from_ids(
            query,
            ids=test_ids,
            is_batched=False,
        )
    )
    assert len(rows) == 5
    ids, nums = zip(*rows, strict=False)
    assert set(ids) == set(test_ids)
    assert set(nums) == {0, 1, 2, 3, 4}


def test_dataset_rows_select_from_ids_batched(numbers_table, warehouse):
    query_ids = (
        sa.select(numbers_table.c.sys__id).order_by(numbers_table.c.num).limit(6)
    )
    test_ids = [r[0] for r in warehouse.db.execute(query_ids)]
    assert len(test_ids) == len(set(test_ids)) == 6

    # Split into two batches: odd and even
    batched_ids = [test_ids[::2], test_ids[1::2]]

    query = sa.select(numbers_table.c.sys__id, numbers_table.c.num)
    batches = list(
        warehouse.dataset_rows_select_from_ids(
            query,
            ids=batched_ids,
            is_batched=True,
        )
    )
    assert len(batches) == 2

    ids, nums = zip(*batches[0], strict=False)
    assert set(ids) == set(batched_ids[0])
    assert set(nums) == {0, 2, 4}

    ids, nums = zip(*batches[1], strict=False)
    assert set(ids) == set(batched_ids[1])
    assert set(nums) == {1, 3, 5}


@pytest.mark.parametrize("is_batched", [True, False])
def test_dataset_rows_select_from_ids_requires_sys_id(
    is_batched, numbers_table, warehouse
):
    # Build a query without sys__id
    query = sa.select(numbers_table.c.num)

    with pytest.raises(RuntimeError, match="sys__id column not found in query"):
        list(
            warehouse.dataset_rows_select_from_ids(
                query,
                ids=[1, 2, 3],
                is_batched=is_batched,
            )
        )
