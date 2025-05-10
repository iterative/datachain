import math
from types import GeneratorType
from typing import Union

import pytest
import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID, partition_columns
from datachain.query.batch import Batch, NoBatching, Partition
from datachain.sql.functions import path


def to_str(val: Union[bytes, str]) -> str:
    """Convert bytes to string if necessary."""
    return val.decode("utf-8") if isinstance(val, bytes) else val


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_no_batching_full_row(warehouse, animal_dataset):
    table = warehouse.get_table(
        warehouse.dataset_table_name(animal_dataset.name, animal_dataset.latest_version)
    )
    cols = (table.c.sys__id, table.c.file__path)
    db_ids, db_files = zip(*warehouse.db.execute(sa.select(*cols)))

    batching = NoBatching()
    rows = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(table.c.sys__id),
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 7
    assert all(len(row) == 2 for row in rows)

    ids, paths = zip(*rows)
    assert set(ids) == set(db_ids)
    assert {to_str(f) for f in paths} == set(db_files)


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_no_batching_ids_only(warehouse, animal_dataset):
    table = warehouse.get_table(
        warehouse.dataset_table_name(animal_dataset.name, animal_dataset.latest_version)
    )
    cols = (table.c.sys__id, table.c.file__path)
    db_ids = [r[0] for r in warehouse.db.execute(sa.select(table.c.sys__id))]

    batching = NoBatching()
    rows = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(table.c.sys__id),
        id_col=table.c.sys__id,
    )
    assert isinstance(rows, GeneratorType)

    rows = list(rows)
    assert len(rows) == 7
    assert set(rows) == set(db_ids)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_batching_full_row(batch_size, warehouse, animal_dataset):
    table = warehouse.get_table(
        warehouse.dataset_table_name(animal_dataset.name, animal_dataset.latest_version)
    )
    cols = (table.c.sys__id, table.c.file__path)
    db_values = set(warehouse.db.execute(sa.select(*cols)))

    batching = Batch(batch_size)
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(table.c.sys__id, table.c.file__path).order_by(table.c.sys__id),
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == math.ceil(7 / batch_size)
    for i, batch in enumerate(batches):
        assert isinstance(batch, list)
        if i < len(batches) - 1:
            assert len(batch) == batch_size
        else:
            assert 0 < len(batch) <= batch_size

        for row in batch:
            row = (row[0], to_str(row[1]))
            assert row in db_values
            db_values.remove(row)

    assert not db_values


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_batching_ids_only(batch_size, warehouse, animal_dataset):
    table = warehouse.get_table(
        warehouse.dataset_table_name(animal_dataset.name, animal_dataset.latest_version)
    )
    cols = (table.c.sys__id, table.c.file__path)
    db_ids = {r[0] for r in warehouse.db.execute(sa.select(table.c.sys__id))}

    batching = Batch(count=batch_size)
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols).order_by(table.c.sys__id),
        id_col=table.c.sys__id,
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == math.ceil(7 / batch_size)
    for i, batch in enumerate(batches):
        assert isinstance(batch, list)
        if i < len(batches) - 1:
            assert len(batch) == batch_size
        else:
            assert 0 < len(batch) <= batch_size

        for row_id in batch:
            assert row_id in db_ids
            db_ids.remove(row_id)

    assert not db_ids


@pytest.fixture
def partitioned_animal_dataset_query(warehouse, animal_dataset):
    table = warehouse.get_table(
        warehouse.dataset_table_name(animal_dataset.name, animal_dataset.latest_version)
    )

    partition_by = [path.parent(table.c.file__path)]

    # create table with partitions
    partition_tbl = warehouse.create_udf_table(partition_columns())

    # fill table with partitions
    cols = [
        table.c.sys__id,
        sa.func.dense_rank().over(order_by=partition_by).label(PARTITION_COLUMN_ID),
    ]
    warehouse.db.execute(partition_tbl.insert().from_select(cols, sa.select(*cols)))

    yield (
        sa.select(*table.columns)
        .outerjoin(partition_tbl, partition_tbl.c.sys__id == table.c.sys__id)
        .add_columns(*partition_columns())
    )

    warehouse.db.drop_table(partition_tbl, if_exists=True)


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_partition_full_row(warehouse, partitioned_animal_dataset_query):
    query = partitioned_animal_dataset_query
    subq = query.subquery()
    cols = (subq.c.sys__id, subq.c.file__path)

    db_ids, db_files = zip(*warehouse.db.execute(sa.select(*cols)))
    db_ids, db_files = set(db_ids), set(db_files)

    partition_files = {
        ("description",),
        ("cats/cat1", "cats/cat2"),
        ("dogs/dog1", "dogs/dog2", "dogs/dog3"),
        ("dogs/others/dog4",),
    }

    batching = Partition()
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols, subq.c.partition_id).order_by(subq.c.sys__id),
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == 4

    for batch in batches:
        assert isinstance(batch, list)

        ids = {row[0] for row in batch}
        assert ids.issubset(db_ids)
        db_ids = db_ids - ids

        files = {to_str(row[1]) for row in batch}
        assert files.issubset(db_files)
        db_files = db_files - files

        files = tuple(sorted(files))
        assert files in partition_files
        partition_files.remove(files)

    assert not db_ids
    assert not db_files
    assert not partition_files


@pytest.mark.parametrize("cloud_type,version_aware", [("file", False)], indirect=True)
def test_partition_ids_only(warehouse, partitioned_animal_dataset_query):
    query = partitioned_animal_dataset_query
    subq = query.subquery()
    cols = (subq.c.sys__id, subq.c.file__path)

    db_rows = {file: id_ for id_, file in warehouse.db.execute(sa.select(*cols))}
    partition_files = {
        ("description",),
        ("cats/cat1", "cats/cat2"),
        ("dogs/dog1", "dogs/dog2", "dogs/dog3"),
        ("dogs/others/dog4",),
    }
    partition_ids = {
        tuple(sorted(db_rows[file] for file in files)) for files in partition_files
    }
    db_ids = set(db_rows.values())

    batching = Partition()
    batches = batching(
        warehouse.dataset_select_paginated,
        sa.select(*cols, subq.c.partition_id).order_by(subq.c.sys__id),
        id_col=subq.c.sys__id,
    )
    assert isinstance(batches, GeneratorType)

    batches = list(batches)
    assert len(batches) == 4

    for batch in batches:
        assert isinstance(batch, list)

        ids = set(batch)
        assert ids.issubset(db_ids)
        db_ids = db_ids - ids

        ids = tuple(sorted(ids))
        assert ids in partition_ids
        partition_ids.remove(ids)

    assert not db_ids
    assert not partition_ids
