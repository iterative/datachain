import io
import json
import math
import os
import pickle
import random
import signal
import subprocess  # nosec B404
import uuid
from datetime import datetime, timedelta, timezone
from json import dumps
from textwrap import dedent
from time import sleep
from unittest.mock import ANY, patch
import posixpath
import uuid
from unittest.mock import ANY

import pytest
import sqlalchemy

from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.error import DatasetInvalidVersionError, DatasetNotFoundError
from datachain.node import Node
from datachain.query import (
    C,
    DatasetQuery,
    DatasetRow,
    LocalFilename,
    Object,
    Stream,
    udf,
)
from datachain.query.builtins import checksum, index_tar
from datachain.query.dataset import QueryStep
from datachain.sql import functions
from datachain.sql.functions.array import cosine_distance, euclidean_distance
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
    SQLType,
    String,
)
from tests.data import ENTRIES
from tests.utils import (
    DEFAULT_TREE,
    LARGE_TREE,
    NUM_TREE,
    SIMPLE_DS_QUERY_RECORDS,
    TARRED_TREE,
    WEBFORMAT_TREE,
    assert_row_names,
    create_tar_dataset,
    dataset_dependency_asdict,
    make_index,
    text_embedding,
)
from datachain.error import (
    DatasetInvalidVersionError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
)
from datachain.query import C, DatasetQuery, Object, Stream
from datachain.sql.functions import path as pathfunc
from datachain.sql.types import String
from tests.utils import assert_row_names, dataset_dependency_asdict

WORKER_COUNT = 1
WORKER_SHUTDOWN_WAIT_SEC = 30


@pytest.fixture()
def run_datachain_worker():
    if not os.environ.get("DATACHAIN_DISTRIBUTED"):
        pytest.skip("Distributed tests are disabled")
    workers = []
    worker_cmd = [
        "celery",
        "-A",
        "datachain_server.distributed",
        "worker",
        "--loglevel=INFO",
        "-P",
        "solo",
        "-Q",
        "datachain-worker",
        "-n",
        "datachain-worker-tests",
    ]
    workers.append(subprocess.Popen(worker_cmd, shell=False))  # noqa: S603
    try:
        from datachain_server.distributed import app

        inspect = app.control.inspect()
        attempts = 0
        # Wait 10 seconds for the Celery worker(s) to be up
        while not inspect.active() and attempts < 10:
            sleep(1)
            attempts += 1

        if attempts == 10:
            raise RuntimeError("Celery worker(s) did not start in time")

        yield workers
    finally:
        for worker in workers:
            os.kill(worker.pid, signal.SIGTERM)
        for worker in workers:
            try:
                worker.wait(timeout=WORKER_SHUTDOWN_WAIT_SEC)
            except subprocess.TimeoutExpired:
                os.kill(worker.pid, signal.SIGKILL)


def from_result_row(col_names, row):
    return dict(zip(col_names, row))


@pytest.fixture
def dogs_cats_dataset(listed_bucket, cloud_test_catalog, dogs_dataset, cats_dataset):
    dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .union(DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog))
        .save(dataset_name)
    )
    return catalog.get_dataset(dataset_name)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_delete_dataset(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=2)

    DatasetQuery.delete("cats", version=1, catalog=catalog)
    dataset = catalog.get_dataset("cats")
    assert dataset.versions_values == [2]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_delete_dataset_latest_version(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=2)

    DatasetQuery.delete("cats", catalog=catalog)
    dataset = catalog.get_dataset("cats")
    assert dataset.versions_values == [1]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_delete_dataset_only_version(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)

    DatasetQuery.delete("cats", catalog=catalog)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("cats")


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_delete_dataset_missing_version(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=2)

    with pytest.raises(DatasetInvalidVersionError):
        DatasetQuery.delete("cats", version=5, catalog=catalog)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_dataset_version_already_exists(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)
    with pytest.raises(RuntimeError) as exc_info:
        DatasetQuery(cats_dataset.name, catalog=catalog).save("cats", version=1)

    assert str(exc_info.value) == "Dataset cats already has version 1"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_multiple_versions(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    # ensure we can select a subset of a bucket properly
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    ds_name = "animals_cats"
    q = ds
    q.save(ds_name)

    q = q.filter(C("file.path").glob("cats*") | (C("file.size") < 4))
    q.save(ds_name)
    q.save(ds_name)

    dataset_record = catalog.get_dataset(ds_name)
    assert dataset_record.status == DatasetStatus.COMPLETE
    assert DatasetQuery(name=ds_name, version=1, catalog=catalog).count() == 7
    assert DatasetQuery(name=ds_name, version=2, catalog=catalog).count() == 3
    assert DatasetQuery(name=ds_name, version=3, catalog=catalog).count() == 3

    with pytest.raises(DatasetVersionNotFoundError):
        DatasetQuery(name=ds_name, version=4, catalog=catalog).count()


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_filter(cloud_test_catalog, save, cats_dataset):
    catalog = cloud_test_catalog.catalog
    # ensure we can select a subset of a bucket properly
    ds = DatasetQuery(cats_dataset.name, catalog=catalog)
    q = (
        ds.filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .filter(C("file.path").regexp("^cats/cat[0-9]$"))
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        q = DatasetQuery(name=ds_name, catalog=catalog)
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    result = q.db_results()
    count = q.count()
    assert len(result) == 2
    assert count == 2


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_instance_returned_after_save(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    ds2 = ds.save("dogs_cats_2")
    assert isinstance(ds2, DatasetQuery)
    expected_names = {"cat1", "cat2", "dog1", "dog2", "dog3", "dog4"}
    assert_row_names(catalog, dogs_cats_dataset, 1, expected_names)
    assert_row_names(catalog, catalog.get_dataset("dogs_cats_2"), 1, expected_names)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_query_specific_dataset_set_proper_dataset_name_version(
    cloud_test_catalog, dogs_cats_dataset
):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)
    assert ds.name == dogs_cats_dataset.name
    assert ds.version == 1


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_set_proper_dataset_name_version(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)
    ds = ds.filter(C("file.path").glob("*dog*"))
    ds2 = ds.save("dogs_small")

    assert ds2.name == "dogs_small"
    assert ds2.version == 1
    assert len(ds2.steps) == 0

    # old dataset query remains detached
    assert ds.name is None
    assert ds.version is None


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_reset_dataset_name_version_after_filter(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)
    ds2 = ds.save("dogs_small")
    assert ds2.name == "dogs_small"
    assert ds2.version == 1

    ds3 = ds2.filter(C("file.path").glob("*dog1"))
    assert ds3.name is None
    assert ds3.version is None

    # old ds2 remains attached
    assert ds2.name == "dogs_small"
    assert ds2.version == 1


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_chain_after_save(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)
    ds.filter(C("file.path").glob("*dog*")).save("ds1").filter(C("file.size") < 4).save(
        "ds2"
    )

    assert_row_names(
        catalog, catalog.get_dataset("ds1"), 1, {"dog1", "dog2", "dog3", "dog4"}
    )
    assert_row_names(catalog, catalog.get_dataset("ds2"), 1, {"dog2"})


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_sselect(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.order_by(C("file.size").desc())
        .limit(6)
        .select(
            C("file.size"), size10x=C("file.size") * 10, size100x=C("file.size") * 100
        )
    )
    result = q.db_results()
    assert result == [
        (13, 130, 1300),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
        (4, 40, 400),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_select_missing_column(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    ds1 = ds.select(C.missing_column_name)
    ds2 = ds.select("missing_column_name")
    # The exception type varies by database backend
    exc1 = pytest.raises(Exception, ds1.db_results)
    assert "missing_column_name" in str(exc1.value)
    exc2 = pytest.raises(KeyError, ds2.db_results)
    assert "missing_column_name" in str(exc2.value)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_select_except(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.order_by(C("file.size").desc())
        .limit(6)
        .select(
            C("file.path"),
            C("file.size"),
            size10x=C("file.size") * 10,
            size100x=C("file.size") * 100,
        )
        .select_except(C("file.path"), C.size10x)
    )
    result = q.db_results()
    assert result == [
        (13, 1300),
        (4, 400),
        (4, 400),
        (4, 400),
        (4, 400),
        (4, 400),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_distinct(cloud_test_catalog, animal_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)

    q = (
        ds.select(pathfunc.name(C("file.path")), C("file.size"))
        .order_by(pathfunc.name(C("file.path")))
        .distinct(C("file.size"))
    )
    result = q.db_results()

    assert result == [("cat1", 4), ("description", 13), ("dog2", 3)]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_distinct_count(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)

    assert ds.distinct(C("file.size")).count() == 3
    assert ds.distinct(C("file.path")).count() == 7
    assert ds.distinct().count() == 7


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_mutate(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.mutate(size10x=C("file.size") * 10)
        .mutate(size1000x=C.size10x * 100)
        .mutate(
            ("s2", C("file.size") * 2),
            ("s3", C("file.size") * 3),
            s4=C("file.size") * 4,
        )
        .filter((C.size10x < 40) | (C.size10x > 100) | C("file.path").glob("cat*"))
        .order_by(C.size10x.desc(), C("file.path"))
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        new_query = DatasetQuery(name=ds_name, catalog=catalog).order_by(
            C.size10x.desc(), C("file.path")
        )
        result = new_query.db_results(row_factory=lambda c, v: dict(zip(c, v)))
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results(row_factory=lambda c, v: dict(zip(c, v)))
    assert len(result) == 4
    assert len(result[0]) == 15
    cols = {"size10x", "size1000x", "s2", "s3", "s4"}
    new_data = [[v for k, v in r.items() if k in cols] for r in result]
    assert new_data == [
        [130, 13000, 26, 39, 52],
        [40, 4000, 8, 12, 16],
        [40, 4000, 8, 12, 16],
        [30, 3000, 6, 9, 12],
    ]


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_order_by_after_mutate(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = (
        ds.mutate(size10x=C("file.size") * 10)
        .filter((C.size10x < 40) | (C.size10x > 100) | C("file.path").glob("cat*"))
        .order_by(C.size10x.desc())
    )

    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = (
            DatasetQuery(name=ds_name, catalog=catalog)
            .order_by(C.size10x.desc(), pathfunc.name(C("file.path")))
            .db_results(row_factory=lambda c, v: dict(zip(c, v)))
        )
    else:
        result = q.db_results(row_factory=lambda c, v: dict(zip(c, v)))

    assert [r["size10x"] for r in result] == [130, 40, 40, 30]


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_order_by_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds = DatasetQuery(animal_dataset.name, catalog=catalog)
    q = ds.order_by(pathfunc.name(C("file.path")).desc()).limit(5)
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        new_query = DatasetQuery(name=ds_name, catalog=catalog).order_by(
            pathfunc.name(C("file.path")).desc()
        )
        result = new_query.db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert [posixpath.basename(r[3]) for r in result] == [
        "dog4",
        "dog3",
        "dog2",
        "dog1",
        "description",
    ]


@pytest.mark.parametrize("save", [True, False])
def test_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .order_by(C("file.path"))
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["cat1", "cat2"]


@pytest.mark.parametrize("save", [True, False])
def test_offset_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .order_by(C("file.path"))
        .offset(3)
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["dog1", "dog2"]


@pytest.mark.parametrize("save", [True, False])
def test_mutate_offset_limit(cloud_test_catalog, save, animal_dataset):
    catalog = cloud_test_catalog.catalog
    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .order_by(C("file.path"))
        .mutate(size10x=C("file.size") * 10)
        .offset(3)
        .limit(2)
    )
    if save:
        ds_name = "animals_cats"
        q.save(ds_name)
        result = DatasetQuery(name=ds_name, catalog=catalog).db_results()
        dataset_record = catalog.get_dataset(ds_name)
        assert dataset_record.status == DatasetStatus.COMPLETE
    else:
        result = q.db_results()

    assert len(result) == 2
    assert [posixpath.basename(r[3]) for r in result] == ["dog1", "dog2"]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_without_explicit_order_by(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).filter(C("file.size") > 0).save(
        ds_name
    )

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    assert len(results) == 7  # unordered, just checking num of results


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_descending(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        pathfunc.name(C("file.path")).desc()
    ).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "dogs/others/dog4"},
        {"sys__id": 2, "file__path": "dogs/dog3"},
        {"sys__id": 3, "file__path": "dogs/dog2"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "description"},
        {"sys__id": 6, "file__path": "cats/cat2"},
        {"sys__id": 7, "file__path": "cats/cat1"},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_ascending(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        pathfunc.name(C("file.path")).asc()
    ).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "cats/cat1"},
        {"sys__id": 2, "file__path": "cats/cat2"},
        {"sys__id": 3, "file__path": "description"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "dogs/dog2"},
        {"sys__id": 6, "file__path": "dogs/dog3"},
        {"sys__id": 7, "file__path": "dogs/others/dog4"},
    ]


def to_str(buf) -> str:
    return io.TextIOWrapper(buf, encoding="utf8").read()


@pytest.mark.parametrize("use_cache", [False, True])
def test_extract(cloud_test_catalog, dogs_dataset, use_cache):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = set()
    for path, stream in q.extract("file__path", Stream(), cache=use_cache):
        with stream:
            value = stream.read().decode("utf-8")
        results.add((posixpath.basename(path), value))
    assert results == {
        ("dog1", "woof"),
        ("dog2", "arf"),
        ("dog3", "bark"),
        ("dog4", "ruff"),
    }


def test_extract_object(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    ds = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    data = ds.extract(Object(to_str), "file__path")
    assert {(value, posixpath.basename(path)) for value, path in data} == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    n = 5
    all_data = []
    ds = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    for i in range(n):
        data = ds.chunk(i, n).extract(Object(to_str), "file__path")
        all_data.extend(data)

    assert {(value, posixpath.basename(path)) for value, path in all_data} == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked_limit(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    chunks = 5
    limit = 1
    all_data = []
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    # Add sufficient rows to ensure each chunk has rows
    for _ in range(5):
        q = q.union(q)
    for i in range(chunks):
        data = q.limit(limit).chunk(i, chunks).extract(Object(to_str), "file__path")
        all_data.extend(data)

    assert len(all_data) == limit


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_limit(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = list(q.limit(2).extract("file__path"))
    assert len(results) == 2


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_order_by(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = list(q.order_by("sys__rand").extract("file__path"))
    pairs = list(q.extract("sys__rand", "file__path"))
    assert results == [(p[1],) for p in sorted(pairs)]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_union(cloud_test_catalog, cats_dataset, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    (dogs | cats).save("dogs_cats")

    q = DatasetQuery(name="dogs_cats", version=1, catalog=catalog)
    result = q.db_results()
    count = q.count()
    assert len(result) == 6
    assert count == 6


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.parametrize("n_columns", [1, 2])
def test_join_with_binary_expression(
    cloud_test_catalog, dogs_dataset, dogs_cats_dataset, inner, n_columns
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    if n_columns == 1:
        predicate = dogs_cats.c("file__path") == dogs.c("file__path")
    else:
        predicate = (dogs_cats.c("file__path") == dogs.c("file__path")) & (
            dogs_cats.c("file__size") == dogs.c("file__size")
        )

    res = dogs_cats.join(
        dogs,
        predicate,
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["file__path_right"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.parametrize("column_predicate", ["file__path", C("file.path")])
def test_join_with_combination_binary_expression_and_column_predicates(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    inner,
    column_predicate,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    res = dogs_cats.join(
        dogs,
        [column_predicate, dogs_cats.c("file__size") == dogs.c("file__size")],
        inner=inner,
    ).to_db_records()

@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("batch", [False, True])
def test_udf_parallel(cloud_test_catalog_tmpfile, batch):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("name",), {"name_len": Int})
    def name_len_local(name):
        # A very simple udf.
        return (len(name),)

    @udf(("name",), {"name_len": Int}, batch=2)
    def name_len_batch(names):
        # A very simple udf.
        return [(len(name),) for (name,) in names]

    if batch:
        # Batching is enabled, we need a udf that acts on
        # lists of inputs.
        udf_func = name_len_batch
    else:
        udf_func = name_len_local

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(udf_func, parallel=-1)
        .select(C.name, C.name_len)
    )
    result = q.db_results()

    assert len(result) == 3
    for r in result:
        # Check that the UDF ran successfully
        assert len(r[0]) == r[1]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("batch", [1, 4])
def test_class_udf_parallel(cloud_test_catalog_tmpfile, batch):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("size",), {"total": Int}, method="sum", batch=batch)
    class MyUDF:
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier
            self.batch = batch

        def sum(self, size):
            if self.batch > 1:
                return [(self.constant + size_ * self.multiplier,) for (size_,) in size]
            return (self.constant + size * self.multiplier,)

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .add_signals(MyUDF(5, multiplier=2), parallel=2)
    )
    results = q.select(C.size, C.total).order_by(C.size).db_results()
    assert results == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf_parallel_exec_error(cloud_test_catalog_tmpfile):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf((C.name,), {"name_len": Int})
    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(name_len_error, parallel=-1)
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        q.db_results()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf_parallel_interrupt(cloud_test_catalog_tmpfile, capfd):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("name",), {"name_len": Int})
    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(name_len_interrupt, parallel=-1)
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        q.db_results()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed(
    cloud_test_catalog_tmpfile,
    batch,
    workers,
    tree,
    datachain_job_id,
    run_datachain_worker,
):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("name",), {"name_len": Int, "blank": String})
    def name_len_local(name):
        # A very simple udf.
        return len(name), None

    @udf(("name",), {"name_len": Int, "blank": String}, batch=2)
    def name_len_batch(names):
        # A very simple udf.
        return [(len(name), None) for (name,) in names]

    if batch:
        # Batching is enabled, we need a udf that acts on lists of inputs.
        udf_func = name_len_batch
    else:
        udf_func = name_len_local

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 90)
        .filter(C.parent.glob("cats*") | (C.size > 30))
        .add_signals(udf_func, parallel=2, workers=workers)
        .select(C.name, C.name_len, C.blank)
    )
    result = q.db_results()

    assert len(result) == 148
    string_default = String.default_value(catalog.warehouse.db.dialect)
    for r in result:
        # Check that the UDF ran successfully
        assert len(r[0]) == r[1]
        assert r[2] == string_default


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_exec_error(
    cloud_test_catalog_tmpfile, workers, datachain_job_id, tree, run_datachain_worker
):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf((C.name,), {"name_len": Int})
    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(name_len_error, parallel=2, workers=workers)
    )
    with pytest.raises(RuntimeError, match="Test Error!"):
        q.db_results()


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_interrupt(
    cloud_test_catalog_tmpfile, capfd, datachain_job_id, tree, run_datachain_worker
):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("name",), {"name_len": Int})
    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(name_len_interrupt, parallel=2, workers=2)
    )
    with pytest.raises(RuntimeError, match=r"Worker Killed \(KeyboardInterrupt\)"):
        q.db_results()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware, tree",
    [("s3", True, LARGE_TREE)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_cancel(
    cloud_test_catalog_tmpfile, capfd, datachain_job_id, tree, run_datachain_worker
):
    catalog = cloud_test_catalog_tmpfile.catalog
    metastore = catalog.metastore
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    job_id = os.environ.get("DATACHAIN_JOB_ID")

    # A job is required for query script cancellation (not using a KeyboardInterrupt)
    metastore.db.execute(
        metastore._jobs_insert().values(
            id=job_id,
            status=7,  # CANCELING
            celery_task_id="",
            name="Test Cancel Job",
            workers=2,
            team_id=metastore.team_id,
            created_at=datetime.now(timezone.utc),
            params="{}",
            metrics="{}",
        ),
    )

    @udf(("name",), {"name_len": Int})
    def name_len_slow(name):
        # A very simple udf, that processes slowly to emulate being stuck.
        from time import sleep

        sleep(10)
        return len(name), None

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C.size < 13)
        .filter(C.parent.glob("cats*") | (C.size < 4))
        .add_signals(name_len_slow, parallel=2, workers=2)
    )

    with pytest.raises(SystemExit) as excinfo:
        q.db_results()

    assert excinfo.value.code == QUERY_SCRIPT_CANCELED_EXIT_CODE
    captured = capfd.readouterr()
    assert "canceled" in captured.out
    assert "semaphore" not in captured.err


def test_apply_udf(cloud_test_catalog, tmp_path):
    catalog = cloud_test_catalog.catalog
    sources = [cloud_test_catalog.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.index(sources)
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    code = """\
        from datachain.query import C, udf
        from datachain.sql.types import Int

        @udf((C.name,), {"name_len": Int})
        def name_len(name):
            # A very simple udf.
            return (len(name),)
    """
    script = tmp_path / "foo.py"
    script.write_text(dedent(code))

    catalog.apply_udf(f"{script}:name_len", cloud_test_catalog.src_uri, "from-storage")
    q = DatasetQuery(name="from-storage", version=1, catalog=catalog).filter(
        C.name_len == 4
    )
    assert len(q.db_results()) == 6

    catalog.apply_udf(f"{script}:name_len", "ds://animals", "from-dataset")
    q = DatasetQuery(name="from-dataset", version=1, catalog=catalog).filter(
        C.name_len == 4
    )
    assert len(q.db_results()) == 6


def to_str(buf) -> str:
    return io.TextIOWrapper(buf, encoding="utf8").read()


@pytest.mark.parametrize("param", [LocalFilename(), Object(to_str)])
@pytest.mark.parametrize("use_cache", [False, True])
def test_udf_object_param(cloud_test_catalog, dogs_dataset, param, use_cache):
    catalog = cloud_test_catalog.catalog
    if isinstance(param, Object):

        @udf((C.name, param), {"signal": String})
        def signal(name, obj):
            # A very simple udf.
            return (name + " -> " + obj,)

    else:

        @udf(("name", param), {"signal": String})
        def signal(name, local_filename):
            with open(local_filename, encoding="utf8") as f:
                obj = f.read()
            return (name + " -> " + obj,)

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signal, cache=use_cache
    )
    result = q.db_results()

    assert len(result) == 4
    signals = {r[-1] for r in result}
    assert signals == {"dog1 -> woof", "dog2 -> arf", "dog3 -> bark", "dog4 -> ruff"}

    uid = Node(*result[0][:-1]).as_uid()
    assert catalog.cache.contains(uid) is (
        use_cache or isinstance(param, LocalFilename)
    )


@pytest.mark.parametrize("use_cache", [False, True])
def test_udf_stream_param(cloud_test_catalog, dogs_dataset, use_cache):
    catalog = cloud_test_catalog.catalog

    @udf((C.name, Stream()), {"signal": String})
    def signal(name, stream):
        with stream as buf:
            return (name + " -> " + buf.read().decode("utf-8"),)

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signal, cache=use_cache
    )
    result = q.db_results()

    assert len(result) == 4
    signals = {r[-1] for r in result}
    assert signals == {"dog1 -> woof", "dog2 -> arf", "dog3 -> bark", "dog4 -> ruff"}

    uid = Node(*result[0][:-1]).as_uid()
    assert catalog.cache.contains(uid) is use_cache


@pytest.mark.parametrize("use_cache", [False, True])
def test_extract(cloud_test_catalog, dogs_dataset, use_cache):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = set()
    for name, stream in q.extract("name", Stream(), cache=use_cache):
        with stream:
            value = stream.read().decode("utf-8")
        results.add((name, value))
    assert results == {
        ("dog1", "woof"),
        ("dog2", "arf"),
        ("dog3", "bark"),
        ("dog4", "ruff"),
    }


def test_extract_object(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    ds = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    data = ds.extract(Object(to_str), "name")
    assert set(data) == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    n = 5
    all_data = []
    ds = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    for i in range(n):
        data = ds.chunk(i, n).extract(Object(to_str), "name")
        all_data.extend(data)

    assert set(all_data) == {
        ("woof", "dog1"),
        ("arf", "dog2"),
        ("bark", "dog3"),
        ("ruff", "dog4"),
    }


def test_extract_chunked_limit(cloud_test_catalog, dogs_dataset):
    ctc = cloud_test_catalog
    chunks = 5
    limit = 1
    all_data = []
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=ctc.catalog)
    # Add sufficient rows to ensure each chunk has rows
    for _ in range(5):
        q = q.union(q)
    for i in range(chunks):
        data = q.limit(limit).chunk(i, chunks).extract(Object(to_str), "name")
        all_data.extend(data)

    assert len(all_data) == limit


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_limit(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = list(q.limit(2).extract("name"))
    assert len(results) == 2


@pytest.mark.parametrize(
    "cloud_type, version_aware",
    [("file", False)],
    indirect=True,
)
def test_extract_order_by(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    results = list(q.order_by("sys__rand").extract("name"))
    pairs = list(q.extract("sys__rand", "name"))
    assert results == [(p[1],) for p in sorted(pairs)]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_union(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog
    sources = [str(cloud_test_catalog.src_uri)]
    catalog.index(sources)

    src = cloud_test_catalog.src_uri
    catalog.create_dataset_from_sources("dogs", [f"{src}/dogs/*"], recursive=True)
    catalog.create_dataset_from_sources("cats", [f"{src}/cats/*"], recursive=True)

    dogs = DatasetQuery(name="dogs", version=1, catalog=catalog)
    cats = DatasetQuery(name="cats", version=1, catalog=catalog)

    (dogs | cats).save("dogs_cats")

    q = DatasetQuery(name="dogs_cats", version=1, catalog=catalog)
    result = q.db_results()
    count = q.count()
    assert len(result) == 6
    assert count == 6


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("predicates", ["name", C.name])
def test_join_left_one_column_predicate(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    predicates,
):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    @udf((), {"sig2": Int})
    def signals2():
        return (2,)

    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).add_signals(signals1)
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals2
    )

    joined_records = dogs_cats.join(dogs, predicates).to_db_records()
    assert len(joined_records) == 6

    cat_records_names = ["cat1", "cat2"]

    dogs_cats_records = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    # rows that found match have both signals
    assert all(
        r["sig1"] == 1 and r["sig2"] == 2
        for r in joined_records
        if r["name"] not in cat_records_names
    )

    int_default = Int.default_value(catalog.warehouse.db.dialect)
    # rows from the left that didn't find match (cats) don't have sig2
    assert all(
        r["sig1"] == 1 and r["sig2"] == int_default
        for r in joined_records
        if r["name"] in cat_records_names
    )
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_cats_records if dr["name"] == r["name"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize(
    "predicates", [["name", "parent"], [C.name, C.parent], ["name", C.parent]]
)
def test_join_left_multiple_column_pedicates(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    predicates,
):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    @udf((), {"sig2": Int})
    def signals2():
        return (2,)

    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).add_signals(signals1)
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals2
    )

    cat_records_names = ["cat1", "cat2"]

    dogs_cats_records = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    joined_records = dogs_cats.join(dogs, predicates).to_db_records()
    assert len(joined_records) == 6

    # rows that found match have both signals
    assert all(
        r["sig1"] == 1 and r["sig2"] == 2
        for r in joined_records
        if r["name"] not in cat_records_names
    )
    int_default = Int.default_value(catalog.warehouse.db.dialect)
    # rows from the left that didn't find match (cats) don't have sig2
    assert all(
        r["sig1"] == 1 and r["sig2"] == int_default
        for r in joined_records
        if r["name"] in cat_records_names
    )
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_cats_records if dr["name"] == r["name"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_binary_expression_on_one_column(
    cloud_test_catalog,
    dogs_dataset,
    cats_dataset,
    inner,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)
    dogs_cats = dogs.union(cats)

    res = dogs_cats.join(
        dogs, dogs_cats.c("name") == dogs.c("name"), inner=inner
    ).to_db_records()

    if inner:
        expected = [
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cat1", string_default),
            ("cat2", string_default),
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]

    assert (
        sorted(((r["name"], r["name_right"]) for r in res), key=lambda x: x[0])
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_binary_expression_on_multiple_columns(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    inner,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    res = dogs_cats.join(
        dogs,
        (
            (dogs_cats.c("name") == dogs.c("name"))
            & (dogs_cats.c("parent") == dogs.c("parent"))
        ),
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cat1", string_default),
            ("cat2", string_default),
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]

    assert (
        sorted(((r["name"], r["name_right"]) for r in res), key=lambda x: x[0])
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
@pytest.mark.parametrize("column_predicate", ["name", C.name])
def test_join_with_combination_binary_expression_and_column_predicates(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
    inner,
    column_predicate,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    res = dogs_cats.join(
        dogs,
        [column_predicate, dogs_cats.c("parent") == dogs.c("parent")],
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cat1", string_default),
            ("cat2", string_default),
            ("dog1", "dog1"),
            ("dog2", "dog2"),
            ("dog3", "dog3"),
            ("dog4", "dog4"),
        ]

    assert (
        sorted(((r["name"], r["name_right"]) for r in res), key=lambda x: x[0])
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_binary_expression_with_arithmetics(
    cloud_test_catalog,
    dogs_dataset,
    cats_dataset,
    inner,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    res = cats.join(
        dogs, cats.c("size") == dogs.c("size") + 1, inner=inner
    ).to_db_records()

    assert sorted(((r["name"], r["name_right"]) for r in res), key=lambda x: x[0]) == [
        ("cat1", "dog2"),
        ("cat2", "dog2"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_conflicting_custom_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    @udf((), {"sig1": Int})
    def signals2():
        return (2,)

    ds1 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals1
    )
    ds2 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals2
    )

    joined_records = ds1.join(ds2, "name").to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig1_right"] == 2 for r in joined_records)

    joined_records = ds1.join(ds2, "name", rname="{name}_dupl").to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig1_dupl"] == 2 for r in joined_records)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_inner(
    cloud_test_catalog,
    dogs_dataset,
    dogs_cats_dataset,
):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    @udf((), {"sig2": Int})
    def signals2():
        return (2,)

    dogs_cats = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).add_signals(signals1)
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals2
    )

    joined_records = dogs_cats.join(dogs, "name", inner=True).to_db_records()
    assert len(joined_records) == 4

    dogs_records = DatasetQuery(
        name=dogs_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig2"] == 2 for r in joined_records)
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["name"] == r["name"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )

    # joining on multiple fields
    joined_records = dogs_cats.join(
        dogs, ["parent", "name"], inner=True
    ).to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig2"] == 2 for r in joined_records)
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["name"] == r["name"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_self(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    dogs_records = DatasetQuery(
        name=dogs_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signals1
    )

    joined_records = dogs.join(dogs, "name").to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig1_right"] == 1 for r in joined_records)
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["name"] == r["name"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_missing_predicates(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf((), {"sig1": Int})
    def signals1():
        return (1,)

    @udf((), {"sig2": Int})
    def signals2():
        return (1,)

    dogs1 = DatasetQuery(
        name=dogs_dataset.name, version=1, catalog=catalog
    ).add_signals(signals1)
    dogs2 = DatasetQuery(
        name=dogs_dataset.name, version=1, catalog=catalog
    ).add_signals(signals2)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, "sig1").to_db_records()
    assert str(excinfo.value) == "Column sig1 was not found in right part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, "sig2").to_db_records()
    assert str(excinfo.value) == "Column sig2 was not found in left part of the join"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_wrong_predicates(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, []).to_db_records()
    assert str(excinfo.value) == "Missing predicates"

    with pytest.raises(TypeError) as excinfo:
        dogs1.join(dogs2, [[]]).to_db_records()
    assert str(excinfo.value) == "Unsupported predicate [] for join expression"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_missing_columns_in_expression(
    cloud_test_catalog, dogs_dataset, cats_dataset
):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("wrong") == dogs2.c("name")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in left part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("name") == dogs2.c("wrong")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in right part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("name") == cats.c("name")).to_db_records()
    assert str(excinfo.value) == (
        "Column name was not found in left or right part of the join"
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_using_functions_in_expression(
    cloud_test_catalog, dogs_dataset, dogs_cats_dataset, inner
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    res = dogs_cats.join(
        dogs,
        (
            sqlalchemy.func.upper(dogs_cats.c("name"))
            == sqlalchemy.func.upper(dogs.c("name"))
        ),
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["file__path_right"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_binary_expression_with_arithmetics(
    cloud_test_catalog,
    dogs_dataset,
    cats_dataset,
    inner,
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    res = cats.join(
        dogs, cats.c("file__size") == dogs.c("file__size") + 1, inner=inner
    ).to_db_records()

    assert sorted(
        ((r["file__path"], r["file__path_right"]) for r in res), key=lambda x: x[0]
    ) == [
        ("cats/cat1", "dogs/dog2"),
        ("cats/cat2", "dogs/dog2"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_wrong_predicates(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, []).to_db_records()
    assert str(excinfo.value) == "Missing predicates"

    with pytest.raises(TypeError) as excinfo:
        dogs1.join(dogs2, [[]]).to_db_records()
    assert str(excinfo.value) == "Unsupported predicate [] for join expression"


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_join_with_missing_columns_in_expression(
    cloud_test_catalog, dogs_dataset, cats_dataset
):
    catalog = cloud_test_catalog.catalog

    dogs1 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs2 = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("wrong") == dogs2.c("file__path")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in left part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("file__path") == dogs2.c("wrong")).to_db_records()
    assert str(excinfo.value) == "Column wrong was not found in right part of the join"

    with pytest.raises(ValueError) as excinfo:
        dogs1.join(dogs2, dogs1.c("file__path") == cats.c("file__path")).to_db_records()
    assert str(excinfo.value) == (
        "Column file__path was not found in left or right part of the join"
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("inner", [True, False])
def test_join_with_using_functions_in_expression(
    cloud_test_catalog, dogs_dataset, dogs_cats_dataset, inner
):
    catalog = cloud_test_catalog.catalog
    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    dogs_cats = DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)

    res = dogs_cats.join(
        dogs,
        (
            sqlalchemy.func.upper(dogs_cats.c("file__path"))
            == sqlalchemy.func.upper(dogs.c("file__path"))
        ),
        inner=inner,
    ).to_db_records()

    if inner:
        expected = [
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]
    else:
        string_default = String.default_value(catalog.warehouse.db.dialect)
        expected = [
            ("cats/cat1", string_default),
            ("cats/cat2", string_default),
            ("dogs/dog1", "dogs/dog1"),
            ("dogs/dog2", "dogs/dog2"),
            ("dogs/dog3", "dogs/dog3"),
            ("dogs/others/dog4", "dogs/others/dog4"),
        ]

    assert (
        sorted(
            ((r["file__path"], r["file__path_right"]) for r in res), key=lambda x: x[0]
        )
        == expected
    )


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_simple_dataset_query(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse
    catalog.create_dataset_from_sources("ds1", [ctc.src_uri], recursive=True)
    DatasetQuery(name="ds1", version=1, catalog=catalog).save("ds2")

    ds_queries = []
    for ds_name in ("ds1", "ds2"):
        ds = metastore.get_dataset(ds_name)
        dr = warehouse.dataset_rows(ds)
        dq = dr.select().order_by(dr.c("path"))
        ds_queries.append(dq)

    ds1, ds2 = (
        [
            {k.name: v for k, v in zip(q.selected_columns, r) if k.name != "sys__id"}
            for r in warehouse.db.execute(q)
        ]
        for q in ds_queries
    )

    # everything except the id field should match
    assert ds1 == ds2
    assert [r["file__path"] for r in ds1] == [
        ("cats/cat1"),
        ("cats/cat2"),
        ("description"),
        ("dogs/dog1"),
        ("dogs/dog2"),
        ("dogs/dog3"),
        ("dogs/others/dog4"),
    ]


def test_aggregate(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    assert q.count() == 4
    assert q.sum(C("file.size")) == 15
    assert q.avg(C("file.size")) == 15 / 4
    assert q.min(C("file.size")) == 3
    assert q.max(C("file.size")) == 4


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_to_db_records(cloud_test_catalog, cats_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    ds = (
        DatasetQuery(cats_dataset.name, catalog=catalog)
        .select(C("file__path"), C("file__size"))
        .order_by(C("file__path"))
    )

    assert ds.to_db_records() == [
        {"file__path": "cats/cat1", "file__size": 4},
        {"file__path": "cats/cat2", "file__size": 4},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_one_storage_as_dependency(
    cloud_test_catalog, listed_bucket, indirect, cats_dataset
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]

    DatasetQuery(cats_dataset.name, catalog=catalog).save(ds_name)

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(
            cats_dataset.name, 1, indirect=indirect
        )
    ] == [
        {
            "id": ANY,
            "type": DatasetDependencyType.STORAGE,
            "name": cloud_test_catalog.src_uri,
            "version": str(1),
            "created_at": listing.created_at,
            "dependencies": [],
        }
    ]


@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_one_registered_dataset_as_dependency(
    cloud_test_catalog, dogs_dataset, indirect
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]

    DatasetQuery(name=dogs_dataset.name, catalog=catalog).save(ds_name)

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "version": str(1),
            "created_at": dogs_dataset.get_version(1).created_at,
            "dependencies": [],
        }
    ]

    if indirect:
        expected[0]["dependencies"] = [
            {
                "id": ANY,
                "type": DatasetDependencyType.STORAGE,
                "name": cloud_test_catalog.src_uri,
                "version": str(1),
                "created_at": listing.created_at,
                "dependencies": [],
            }
        ]

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(ds_name, 1, indirect=indirect)
    ] == expected

    catalog.remove_dataset(dogs_dataset.name, force=True)
    # None means dependency was there but was removed in the meantime
    assert catalog.get_dataset_dependencies(ds_name, 1) == [None]


@pytest.mark.parametrize("method", ["union", "join"])
def test_dataset_dependencies_multiple_direct_dataset_dependencies(
    cloud_test_catalog, dogs_dataset, cats_dataset, method
):
    # multiple direct dataset dependencies can be achieved with methods that are
    # combining multiple DatasetQuery instances into new one like union or join
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    if method == "union":
        dogs.union(cats).save(ds_name)
    else:
        dogs.join(cats, "file__path").save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": cloud_test_catalog.src_uri,
        "version": str(1),
        "created_at": listing.created_at,
        "dependencies": [],
    }

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "version": str(1),
            "created_at": dogs_dataset.get_version(1).created_at,
            "dependencies": [storage_depenedncy],
        },
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": cats_dataset.name,
            "version": str(1),
            "created_at": cats_dataset.get_version(1).created_at,
            "dependencies": [storage_depenedncy],
        },
    ]

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, 1, indirect=True)
        ),
        key=lambda d: d["name"],
    ) == sorted(expected, key=lambda d: d["name"])

    # check when removing one dependency
    catalog.remove_dataset(dogs_dataset.name, force=True)
    expected[0] = None
    expected[1]["dependencies"] = []

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, 1)
        ),
        key=lambda d: d["name"] if d else "",
    ) == sorted(expected, key=lambda d: d["name"] if d else "")

    # check when removing the other dependency
    catalog.remove_dataset(cats_dataset.name, force=True)
    assert catalog.get_dataset_dependencies(ds_name, 1) == [None, None]


def test_dataset_dependencies_multiple_union(
    cloud_test_catalog, dogs_dataset, cats_dataset
):
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    listing = catalog.listings()[0]

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)
    dogs_other = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)

    dogs.union(cats).union(dogs_other).save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": cloud_test_catalog.src_uri,
        "version": str(1),
        "created_at": listing.created_at,
        "dependencies": [],
    }

    expected = [
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": dogs_dataset.name,
            "version": str(1),
            "created_at": dogs_dataset.get_version(1).created_at,
            "dependencies": [storage_depenedncy],
        },
        {
            "id": ANY,
            "type": DatasetDependencyType.DATASET,
            "name": cats_dataset.name,
            "version": str(1),
            "created_at": cats_dataset.get_version(1).created_at,
            "dependencies": [storage_depenedncy],
        },
    ]

    assert sorted(
        (
            dataset_dependency_asdict(d)
            for d in catalog.get_dataset_dependencies(ds_name, 1, indirect=True)
        ),
        key=lambda d: d["name"],
    ) == sorted(expected, key=lambda d: d["name"])


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_save_subset_of_columns(cloud_test_catalog, cats_dataset):
    catalog = cloud_test_catalog.catalog
    DatasetQuery(cats_dataset.name, catalog=catalog).select(C("file.path")).save(
        "cats", version=1
    )

    dataset = catalog.get_dataset("cats")
    assert dataset.schema == {"file__path": String}
