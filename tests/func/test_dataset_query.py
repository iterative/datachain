import io
import json
import math
import os
import pickle
import posixpath
import uuid
from datetime import datetime, timezone
from json import dumps
from typing import Any, Optional
from unittest.mock import ANY

import numpy as np
import pytest
import sqlalchemy
from dateutil.parser import isoparse

from datachain.catalog import QUERY_SCRIPT_CANCELED_EXIT_CODE
from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.error import DatasetInvalidVersionError, DatasetNotFoundError
from datachain.lib.file import File
from datachain.query import (
    C,
    DatasetQuery,
    LocalFilename,
    Object,
    Stream,
    udf,
)
from datachain.query.builtins import checksum, index_tar
from datachain.query.dataset import QueryStep
from datachain.query.schema import file_signals
from datachain.sql import functions
from datachain.sql.functions import path as pathfunc
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
from tests.utils import (
    NUM_TREE,
    TARRED_TREE,
    WEBFORMAT_TREE,
    assert_row_names,
    dataset_dependency_asdict,
    text_embedding,
)

FILE_SCHEMA = {
    f"file__{name}": _type if _type != Int else Int64
    for name, _type in File._datachain_column_types.items()
}


def create_row(
    path: str,
    source: str = "",
    size: int = 0,
    location: Optional[dict[str, Any]] = None,
    vtype: str = "",
    is_latest: bool = True,
    last_modified: Optional[datetime] = None,
    version: str = "",
    etag: str = "",
    prefix: str = "file",
):
    if location:
        location = json.dumps([location])  # type: ignore [assignment]

    last_modified = last_modified or datetime.now(timezone.utc)

    return (  # type: ignore [return-value]
        source,
        path,
        size,
        version,
        etag,
        is_latest,
        last_modified,
        location,
        vtype,
    )


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

    with pytest.raises(ValueError):
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
    [("file", False)],
    indirect=True,
)
def test_exec(cloud_test_catalog, dogs_cats_dataset):
    catalog = cloud_test_catalog.catalog
    all_names = set()

    @udf(params=("file__path",), output={})
    def name_len(path):
        all_names.add(posixpath.basename(path))

    existing_datasets = list(catalog.ls_datasets())
    dq = (
        DatasetQuery(name=dogs_cats_dataset.name, version=1, catalog=catalog)
        .add_signals(name_len)
        .exec()
    )
    assert isinstance(dq, DatasetQuery)
    assert all_names == {"dog1", "dog2", "dog3", "dog4", "cat1", "cat2"}
    # exec should not leave any datasets behind
    assert list(catalog.ls_datasets()) == existing_datasets


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
def test_avoid_recalculation_after_save(cloud_test_catalog, animal_dataset):
    calls = 0

    @udf(("file__size",), {"name_len": Int})
    def name_len(size):
        nonlocal calls
        calls += 1
        return (size,)

    catalog = cloud_test_catalog.catalog
    ds = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .filter(C("file.path").glob("*/dog1"))
        .add_signals(name_len)
    )
    ds2 = ds.save("ds1")

    assert ds2.steps == []
    assert ds2.dependencies == set()
    assert isinstance(ds2.starting_step, QueryStep)
    ds2.save("ds2")
    assert calls == 1  # UDF should be called only once


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
def test_select(cloud_test_catalog, animal_dataset):
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


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_len_desc_and_name_asc(
    cloud_test_catalog, animal_dataset
):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    DatasetQuery(animal_dataset.name, catalog=catalog).add_signals(name_len).order_by(
        C.name_len.desc(), pathfunc.name(C("file.path")).asc()
    ).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "description"},
        {"sys__id": 2, "file__path": "cats/cat1"},
        {"sys__id": 3, "file__path": "cats/cat2"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "dogs/dog2"},
        {"sys__id": 6, "file__path": "dogs/dog3"},
        {"sys__id": 7, "file__path": "dogs/others/dog4"},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_before_add_signals(
    cloud_test_catalog, animal_dataset
):
    catalog = cloud_test_catalog.catalog
    ds_name = uuid.uuid4().hex

    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    DatasetQuery(animal_dataset.name, catalog=catalog).order_by(
        pathfunc.name(C("file.path")).asc()
    ).add_signals(name_len).save(ds_name)

    results = DatasetQuery(name=ds_name, catalog=catalog).to_db_records()
    results_name_id = [
        {k: v for k, v in r.items() if k in ["sys__id", "file__path"]} for r in results
    ]
    # we should preserve order in final result based on order by which was added
    # before add_signals
    assert sorted(results_name_id, key=lambda k: k["sys__id"]) == [
        {"sys__id": 1, "file__path": "cats/cat1"},
        {"sys__id": 2, "file__path": "cats/cat2"},
        {"sys__id": 3, "file__path": "description"},
        {"sys__id": 4, "file__path": "dogs/dog1"},
        {"sys__id": 5, "file__path": "dogs/dog2"},
        {"sys__id": 6, "file__path": "dogs/dog3"},
        {"sys__id": 7, "file__path": "dogs/others/dog4"},
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    q = (
        DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len)
    )
    result1 = q.select(C("file.path"), C.name_len).db_results()
    # ensure that we're able to run with same query multiple times
    result2 = q.select(C("file.path"), C.name_len).db_results()
    count = q.count()
    assert len(result1) == 3
    assert len(result2) == 3
    assert count == 3

    for r1, r2 in zip(result1, result2):
        # Check that the UDF ran successfully
        assert len(posixpath.basename(r1[0])) == r1[1]
        assert len(posixpath.basename(r2[0])) == r2[1]

    q.save("test_udf")
    dataset = catalog.get_dataset("test_udf")
    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    expected_schema = FILE_SCHEMA | sys_schema | {"name_len": Int}
    assert dataset.schema == expected_schema


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf_different_types(cloud_test_catalog, animal_dataset):
    catalog = cloud_test_catalog.catalog

    obj = {"name": "John", "age": 30}

    @udf(
        (),
        {
            "int_col": Int,
            "int_col_32": Int32,
            "int_col_64": Int64,
            "float_col": Float,
            "float_col_32": Float32,
            "float_col_64": Float64,
            "array_col": Array(Float),
            "array_col_nested": Array(Array(Float)),
            "array_col_32": Array(Float32),
            "array_col_64": Array(Float64),
            "string_col": String,
            "bool_col": Boolean,
            "json_col": JSON,
            "binary_col": Binary,
        },
    )
    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            dumps({"a": 1}),
            pickle.dumps(obj),
        )

    q = (
        DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog)
        .filter(pathfunc.name(C("file.path")) == "cat1")
        .add_signals(test_types)
    )

    results = q.select().to_db_records()
    col_values = [
        (
            r["int_col"],
            r["int_col_32"],
            r["int_col_64"],
            r["float_col"],
            r["float_col_32"],
            r["float_col_64"],
            r["array_col"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
            r["string_col"],
            r["bool_col"],
            r["json_col"],
            pickle.loads(r["binary_col"]),  # noqa: S301
        )
        for r in results
    ]

    assert col_values == [
        (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            dumps({"a": 1}),
            obj,
        )
    ]

    q.save("test_udf")
    dataset = catalog.get_dataset("test_udf")

    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    expected_schema = (
        FILE_SCHEMA
        | sys_schema
        | {
            "int_col": Int,
            "int_col_32": Int32,
            "int_col_64": Int64,
            "float_col": Float,
            "float_col_32": Float32,
            "float_col_64": Float64,
            "array_col": Array(Float()),
            "array_col_nested": Array(Array(Float())),
            "array_col_32": Array(Float32()),
            "array_col_64": Array(Float64()),
            "string_col": String,
            "bool_col": Boolean,
            "json_col": JSON,
            "binary_col": Binary,
        }
    )

    for c_name, c_type in dataset.schema.items():
        assert c_name in expected_schema
        c_type_expected = expected_schema[c_name]
        if not isinstance(c_type, SQLType):
            c_type = c_type()
            c_type_expected = c_type_expected()

        assert c_type.to_dict() == c_type_expected.to_dict()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("batch", [1, 4])
def test_class_udf(cloud_test_catalog, batch, animal_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__size",), {"total": Int}, method="sum", batch=batch)
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
        DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .add_signals(MyUDF(5, multiplier=2))
    )
    results = q.select(C("file.size"), C.total).order_by(C("file.size")).db_results()
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
def test_udf_reuse_on_error(cloud_test_catalog_tmpfile):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    error_state = {"error": True}

    @udf((C("file.path"),), {"path_len": Int})
    def name_len_maybe_error(path):
        if error_state["error"]:
            # A udf that raises an exception
            raise RuntimeError("Test Error!")
        return (len(path),)

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len_maybe_error)
        .select(C("file.path"), C.path_len)
    )
    with pytest.raises(RuntimeError, match="Test Error!"):
        q.db_results()

    # Simulate fixing the error
    error_state["error"] = False

    # Retry Query
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
@pytest.mark.parametrize("batch", [False, True])
def test_udf_parallel(cloud_test_catalog_tmpfile, batch):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("file__path",), {"name_len": Int})
    def name_len_local(name):
        # A very simple udf.
        return (len(name),)

    @udf(("file__path",), {"name_len": Int}, batch=2)
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
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(udf_func, parallel=-1)
        .select(C("file.path"), C.name_len)
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
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("file__size",), {"total": Int}, method="sum", batch=batch)
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
        .filter(C("file.size") < 13)
        .add_signals(MyUDF(5, multiplier=2), parallel=2)
    )
    results = q.select(C("file.size"), C.total).order_by(C("file.size")).db_results()
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
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf((C("file.path"),), {"name_len": Int})
    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
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
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("file__path",), {"name_len": Int})
    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len_interrupt, parallel=-1)
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        q.db_results()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed(cloud_test_catalog_tmpfile, batch, workers, datachain_job_id):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("file__path",), {"name_len": Int, "blank": String})
    def name_len_local(name):
        # A very simple udf.
        return len(name), None

    @udf(("file__path",), {"name_len": Int, "blank": String}, batch=2)
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
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(udf_func, parallel=2, workers=workers)
        .select(C("file.path"), C.name_len, C.blank)
    )
    result = q.db_results()

    assert len(result) == 3
    string_default = String.default_value(catalog.warehouse.db.dialect)
    for r in result:
        # Check that the UDF ran successfully
        assert len(r[0]) == r[1]
        assert r[2] == string_default


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_exec_error(
    cloud_test_catalog_tmpfile, workers, datachain_job_id
):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf((C("file.path"),), {"name_len": Int})
    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len_error, parallel=2, workers=workers)
    )
    with pytest.raises(RuntimeError, match="Test Error!"):
        q.db_results()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_interrupt(cloud_test_catalog_tmpfile, capfd, datachain_job_id):
    catalog = cloud_test_catalog_tmpfile.catalog
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
    catalog.create_dataset_from_sources("animals", globs, recursive=True)

    @udf(("file__path",), {"name_len": Int})
    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len_interrupt, parallel=2, workers=2)
    )
    with pytest.raises(RuntimeError, match=r"Worker Killed \(KeyboardInterrupt\)"):
        q.db_results()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
def test_udf_distributed_cancel(cloud_test_catalog_tmpfile, capfd, datachain_job_id):
    catalog = cloud_test_catalog_tmpfile.catalog
    metastore = catalog.metastore
    sources = [cloud_test_catalog_tmpfile.src_uri]
    globs = [s.rstrip("/") + "/*" for s in sources]
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

    @udf(("file__path",), {"name_len": Int})
    def name_len_slow(name):
        # A very simple udf, that processes slowly to emulate being stuck.
        from time import sleep

        sleep(10)
        return len(name), None

    q = (
        DatasetQuery(name="animals", version=1, catalog=catalog)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .add_signals(name_len_slow, parallel=2, workers=2)
    )

    with pytest.raises(SystemExit) as excinfo:
        q.db_results()

    assert excinfo.value.code == QUERY_SCRIPT_CANCELED_EXIT_CODE
    captured = capfd.readouterr()
    assert "canceled" in captured.out
    assert "semaphore" not in captured.err


def to_str(buf) -> str:
    return io.TextIOWrapper(buf, encoding="utf8").read()


@pytest.mark.parametrize("param", [LocalFilename(), Object(to_str)])
@pytest.mark.parametrize("use_cache", [False, True])
def test_udf_object_param(cloud_test_catalog, dogs_dataset, param, use_cache):
    catalog = cloud_test_catalog.catalog
    if isinstance(param, Object):

        @udf((C("file.path"), param), {"signal": String})
        def signal(path, obj):
            # A very simple udf.
            return (posixpath.basename(path) + " -> " + obj,)

    else:

        @udf(("file__path", param), {"signal": String})
        def signal(path, local_filename):
            with open(local_filename, encoding="utf8") as f:
                obj = f.read()
            return (posixpath.basename(path) + " -> " + obj,)

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signal, cache=use_cache
    )
    result = q.db_results(row_factory=from_result_row)

    assert len(result) == 4
    signals = {r["signal"] for r in result}
    assert signals == {"dog1 -> woof", "dog2 -> arf", "dog3 -> bark", "dog4 -> ruff"}

    uid = catalog._get_row_uid(file_signals(result[0]))
    assert catalog.cache.contains(uid) is (
        use_cache or isinstance(param, LocalFilename)
    )


@pytest.mark.parametrize("use_cache", [False, True])
def test_udf_stream_param(cloud_test_catalog, dogs_dataset, use_cache):
    catalog = cloud_test_catalog.catalog

    @udf((C("file.path"), Stream()), {"signal": String})
    def signal(path, stream):
        with stream as buf:
            return (posixpath.basename(path) + " -> " + buf.read().decode("utf-8"),)

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        signal, cache=use_cache
    )
    result = q.db_results(row_factory=from_result_row)

    assert len(result) == 4
    signals = {r["signal"] for r in result}
    assert signals == {"dog1 -> woof", "dog2 -> arf", "dog3 -> bark", "dog4 -> ruff"}

    uid = catalog._get_row_uid(file_signals(result[0]))
    assert catalog.cache.contains(uid) is use_cache


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
@pytest.mark.parametrize("predicates", ["file__path", C("file.path")])
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

    cat_records_names = ["cats/cat1", "cats/cat2"]

    dogs_cats_records = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    # rows that found match have both signals
    assert all(
        r["sig1"] == 1 and r["sig2"] == 2
        for r in joined_records
        if r["file__path"] not in cat_records_names
    )

    int_default = Int.default_value(catalog.warehouse.db.dialect)
    # rows from the left that didn't find match (cats) don't have sig2
    assert all(
        r["sig1"] == 1 and r["sig2"] == int_default
        for r in joined_records
        if r["file__path"] in cat_records_names
    )
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(
            dr for dr in dogs_cats_records if dr["file__path"] == r["file__path"]
        )
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize(
    "predicates",
    [
        ["file__path", "file__size"],
        [C("file.path"), C("file.size")],
        ["file__path", C("file.size")],
    ],
)
def test_join_left_multiple_column_predicates(
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

    cat_records_names = ["cats/cat1", "cats/cat2"]

    dogs_cats_records = DatasetQuery(
        name=dogs_cats_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    joined_records = dogs_cats.join(dogs, predicates).to_db_records()
    assert len(joined_records) == 6

    # rows that found match have both signals
    assert all(
        r["sig1"] == 1 and r["sig2"] == 2
        for r in joined_records
        if r["file__path"] not in cat_records_names
    )
    int_default = Int.default_value(catalog.warehouse.db.dialect)
    # rows from the left that didn't find match (cats) don't have sig2
    assert all(
        r["sig1"] == 1 and r["sig2"] == int_default
        for r in joined_records
        if r["file__path"] in cat_records_names
    )
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(
            dr for dr in dogs_cats_records if dr["file__path"] == r["file__path"]
        )
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )


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

    joined_records = ds1.join(ds2, "file__path").to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig1_right"] == 2 for r in joined_records)

    joined_records = ds1.join(ds2, "file__path", rname="{name}_dupl").to_db_records()
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

    joined_records = dogs_cats.join(dogs, "file__path", inner=True).to_db_records()
    assert len(joined_records) == 4

    dogs_records = DatasetQuery(
        name=dogs_dataset.name, version=1, catalog=catalog
    ).to_db_records()

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig2"] == 2 for r in joined_records)
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["file__path"] == r["file__path"])
        assert all(
            [r[f"{k}_right"] == dog_r[k]] for k in dog_r if not k.startswith("sys__")
        )

    # joining on multiple fields
    joined_records = dogs_cats.join(
        dogs, ["file__path", "file__size"], inner=True
    ).to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig2"] == 2 for r in joined_records)
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["file__path"] == r["file__path"])
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

    joined_records = dogs.join(dogs, "file__path").to_db_records()
    assert len(joined_records) == 4

    # check custom columns
    assert all(r["sig1"] == 1 and r["sig1_right"] == 1 for r in joined_records)
    # check core duplicated columns
    for r in joined_records:
        dog_r = next(dr for dr in dogs_records if dr["file__path"] == r["file__path"])
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


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), FILE_SCHEMA)
    def gen(path):
        # A very simple file row generator.
        from datachain.lib.file import File

        yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
        yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(gen)
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
        "dogs/dog2/subobject",
        "dogs/dog2/subobject2",
        "dogs/dog3/subobject",
        "dogs/dog3/subobject2",
        "dogs/others/dog4/subobject",
        "dogs/others/dog4/subobject2",
    ]

    q.save("test_generator")
    dataset = catalog.get_dataset("test_generator")
    schema = dataset.schema
    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    assert schema == FILE_SCHEMA | sys_schema


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_filter(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), FILE_SCHEMA)
    def gen(path):
        # A very simple file row generator.
        from datachain.lib.file import File

        yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
        yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .generate(gen)
        .filter(pathfunc.name(C("file.path")) == "subobject")
    )
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog2/subobject",
        "dogs/dog3/subobject",
        "dogs/others/dog4/subobject",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_limit(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), FILE_SCHEMA)
    def gen(path):
        # A very simple file row generator.
        from datachain.lib.file import File

        yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
        yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .order_by(C("file.path"))
        .limit(1)
        .generate(gen)
    )
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_parallel(cloud_test_catalog_tmpfile):
    # Setup catalog.
    dogs_dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    dogs_dataset = catalog.create_dataset_from_sources(
        dogs_dataset_name, [f"{src_uri}/dogs/*"], recursive=True
    )

    @udf(("file__path",), FILE_SCHEMA)
    def gen(path):
        # A very simple file row generator.
        from datachain.lib.file import File

        yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
        yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        gen, parallel=-1
    )
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
        "dogs/dog2/subobject",
        "dogs/dog2/subobject2",
        "dogs/dog3/subobject",
        "dogs/dog3/subobject2",
        "dogs/others/dog4/subobject",
        "dogs/others/dog4/subobject2",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_batch(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path"), FILE_SCHEMA, batch=4)
    def gen(inputs):
        from datachain.lib.file import File

        for (path,) in inputs:
            yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
            yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(gen)
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
        "dogs/dog2/subobject",
        "dogs/dog2/subobject2",
        "dogs/dog3/subobject",
        "dogs/dog3/subobject2",
        "dogs/others/dog4/subobject",
        "dogs/others/dog4/subobject2",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_class(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(
        params=(C("file.path"),),
        output=FILE_SCHEMA,
        method="generate_subobjects",
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, path):
            from datachain.lib.file import File

            yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
            yield tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70)))

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        Subobjects
    )
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
        "dogs/dog2/subobject",
        "dogs/dog2/subobject2",
        "dogs/dog3/subobject",
        "dogs/dog3/subobject2",
        "dogs/others/dog4/subobject",
        "dogs/others/dog4/subobject2",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_class_batch(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(
        params=(C("file.path"),),
        output=FILE_SCHEMA,
        method="generate_subobjects",
        batch=4,
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, inputs):
            from datachain.lib.file import File

            for (path,) in inputs:
                yield tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
                yield tuple(
                    f[1] for f in list(File(path=f"{path}/subobject2", size=70))
                )

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        Subobjects
    )
    result = q.to_db_records()
    parents_names = sorted(r["file__path"] for r in result)
    assert parents_names == [
        "dogs/dog1/subobject",
        "dogs/dog1/subobject2",
        "dogs/dog2/subobject",
        "dogs/dog2/subobject2",
        "dogs/dog3/subobject",
        "dogs/dog3/subobject2",
        "dogs/others/dog4/subobject",
        "dogs/others/dog4/subobject2",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_partition_by(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), FILE_SCHEMA | {"cnt": Int})
    def gen(inputs):
        from datachain.lib.file import File

        cnt = len(inputs)
        for (path,) in inputs:
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                cnt,
            )
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70))),
                cnt,
            )

    result = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .generate(gen, partition_by=pathfunc.parent(C("file.path")))
        .to_db_records()
    )
    parents_names = [(r["file__path"], r["cnt"]) for r in result]
    parents_names.sort(key=lambda x: (x[0]))
    assert parents_names == [
        ("dogs/dog1/subobject", 3),
        ("dogs/dog1/subobject2", 3),
        ("dogs/dog2/subobject", 3),
        ("dogs/dog2/subobject2", 3),
        ("dogs/dog3/subobject", 3),
        ("dogs/dog3/subobject2", 3),
        ("dogs/others/dog4/subobject", 1),
        ("dogs/others/dog4/subobject2", 1),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_partition_by_parallel(cloud_test_catalog_tmpfile):
    # Setup catalog.
    dogs_dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog_tmpfile.catalog
    src_uri = cloud_test_catalog_tmpfile.src_uri

    dogs_dataset = catalog.create_dataset_from_sources(
        dogs_dataset_name, [f"{src_uri}/dogs/*"], recursive=True
    )

    @udf(("file__path",), FILE_SCHEMA | {"cnt": Int})
    def gen(inputs):
        from datachain.lib.file import File

        cnt = len(inputs)
        for (path,) in inputs:
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                cnt,
            )
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70))),
                cnt,
            )

    result = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .generate(gen, partition_by=pathfunc.parent(C("file.path")), parallel=-1)
        .to_db_records()
    )
    parents_names = [(r["file__path"], r["cnt"]) for r in result]
    parents_names.sort(key=lambda x: (x[0]))
    assert parents_names == [
        ("dogs/dog1/subobject", 3),
        ("dogs/dog1/subobject2", 3),
        ("dogs/dog2/subobject", 3),
        ("dogs/dog2/subobject2", 3),
        ("dogs/dog3/subobject", 3),
        ("dogs/dog3/subobject2", 3),
        ("dogs/others/dog4/subobject", 1),
        ("dogs/others/dog4/subobject2", 1),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_partition_by_batch(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), FILE_SCHEMA | {"cnt": Int}, batch=2)
    def gen(inputs):
        from datachain.lib.file import File

        cnt = len(inputs)
        for (path,) in inputs:
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                cnt,
            )
            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject2", size=70))),
                cnt,
            )

    result = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .generate(gen, partition_by=pathfunc.parent(C("file.path")))
        .to_db_records()
    )
    parents_names = [(r["file__path"], r["cnt"]) for r in result]
    parents_names.sort(key=lambda x: (x[0]))
    assert parents_names == [
        ("dogs/dog1/subobject", 3),
        ("dogs/dog1/subobject2", 3),
        ("dogs/dog2/subobject", 3),
        ("dogs/dog2/subobject2", 3),
        ("dogs/dog3/subobject", 3),
        ("dogs/dog3/subobject2", 3),
        ("dogs/others/dog4/subobject", 1),
        ("dogs/others/dog4/subobject2", 1),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_new_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    now = datetime.now(timezone.utc).replace(microsecond=0)

    int_example = 25

    new_columns = {
        "string_col": String,
        "int_col": Int,
        "int_col_32": Int32,
        "int_col_64": Int64,
        "bool_col": Boolean,
        "float_col": Float,
        "float_col_32": Float32,
        "float_col_64": Float64,
        "json_col": JSON,
        "datetime_col": DateTime,
        "binary_col": Binary,
        "array_col": Array(Float),
        "array_col_nested": Array(Array(Float)),
        "array_col_32": Array(Float32),
        "array_col_64": Array(Float64),
    }

    @udf(
        params=(C("file.path"),),
        output=FILE_SCHEMA | new_columns,
        method="generate_subobjects",
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, path):
            from datachain.lib.file import File

            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                "some_string",
                10,
                11,
                12,
                True,
                0.5,
                0.5,
                0.5,
                dumps({"a": 1}),
                now,
                int_example.to_bytes(2, "big"),
                [0.5, 0.5],
                [[0.5], [0.5]],
                [0.5, 0.5],
                [0.5, 0.5],
            )

    DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        Subobjects
    ).save("dogs_with_rows_and_signals")

    q = DatasetQuery(name="dogs_with_rows_and_signals", catalog=catalog)
    result = q.select().to_db_records()

    col_values = [
        (
            r["file__path"],
            r["string_col"],
            r["int_col"],
            r["int_col_32"],
            r["int_col_64"],
            r["bool_col"],
            r["float_col"],
            r["float_col_32"],
            r["float_col_64"],
            r["json_col"],
            r["datetime_col"].astimezone(timezone.utc) if r["datetime_col"] else None,
            int.from_bytes(r["binary_col"], "big"),  # converting from binary to int
            r["array_col"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
        )
        for r in result
    ]

    col_values.sort(key=lambda x: (x[1], x[0]))

    new_col_values = (
        "some_string",
        10,
        11,
        12,
        True,
        0.5,
        0.5,
        0.5,
        dumps({"a": 1}),
        now,
        int_example,
        [0.5, 0.5],
        [[0.5], [0.5]],
        [0.5, 0.5],
        [0.5, 0.5],
    )

    assert col_values == [
        ("dogs/dog1/subobject", *new_col_values),
        ("dogs/dog2/subobject", *new_col_values),
        ("dogs/dog3/subobject", *new_col_values),
        ("dogs/others/dog4/subobject", *new_col_values),
    ]

    dataset = catalog.get_dataset("dogs_with_rows_and_signals")
    expected_schema = FILE_SCHEMA | new_columns

    dr = catalog.warehouse.schema.dataset_row_cls
    schema = dataset.schema
    assert all(isinstance(c.type, schema.pop(c.name)) for c in dr.sys_columns())

    for c_name, c_type in schema.items():
        assert c_name in expected_schema
        c_type_expected = expected_schema[c_name]
        if not isinstance(c_type, SQLType):
            c_type = c_type()
            c_type_expected = c_type_expected()

        assert c_type.to_dict() == c_type_expected.to_dict()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generators_sequence_with_new_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(params="file__path", output=FILE_SCHEMA | {"path_upper": String, "p1": Int})
    def upper(path):
        yield *tuple(f[1] for f in list(File(path=path))), path.upper(), 1

    @udf(params="file__path", output=FILE_SCHEMA | {"path_lower": String, "p2": Int})
    def lower(path):
        yield *tuple(f[1] for f in list(File(path=path))), path.lower(), 2

    DatasetQuery(name=dogs_dataset.name, catalog=catalog).generate(upper).save("upper")
    for res in DatasetQuery(name="upper", catalog=catalog).to_db_records():
        assert "path_upper" in res
        assert res["path_upper"] == res["file__path"].upper()
        assert "p1" in res
        assert res["p1"] == 1

    DatasetQuery(name="upper", catalog=catalog).generate(lower).save("lower")
    for res in DatasetQuery(name="lower", catalog=catalog).to_db_records():
        assert "path_upper" not in res
        assert "path_lower" in res
        assert res["path_lower"] == res["file__path"].lower()
        assert "p1" not in res
        assert "p2" in res
        assert res["p2"] == 2


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_new_columns_empty_values(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    dialect = catalog.warehouse.db.dialect

    new_columns = {
        "int_col": Int,
        "int_col_32": Int32,
        "int_col_64": Int64,
        "bool_col": Boolean,
        "float_col": Float,
        "float_col_32": Float32,
        "float_col_64": Float64,
        "json_col": JSON,
        "datetime_col": DateTime,
        "binary_col": Binary,
        "array_col": Array(Float),
        "array_col_nested": Array(Array(Float)),
        "array_col_32": Array(Float32),
        "array_col_64": Array(Float64),
    }
    new_col_values_empty = tuple(t.default_value(dialect) for t in new_columns.values())

    @udf(
        params=(C("file.path"),),
        output=FILE_SCHEMA | new_columns,
        method="generate_subobjects",
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, path):
            from datachain.lib.file import File

            yield (
                tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50)))
                + new_col_values_empty
            )

    DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        Subobjects
    ).save("dogs_with_rows_and_signals")

    q = DatasetQuery(name="dogs_with_rows_and_signals", catalog=catalog)
    result = q.to_db_records()

    for row in result:
        for i, col in enumerate(new_columns):
            val = row[col]
            expected = new_col_values_empty[i]
            if isinstance(expected, float) and math.isnan(expected):
                assert math.isnan(val)
            else:
                assert val == expected


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_new_columns_numpy(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    new_columns = {
        "int_col_32": Int32,
        "int_col_64": Int64,
        "float_col_32": Float32,
        "float_col_64": Float64,
        "int_float_col_32": Float32,
        "array_col_nested": Array(Array(Float32)),
        "array_col_32": Array(Float32),
        "array_col_64": Array(Float64),
        "array_int_float_col_32": Array(Float32),
        "array_empty_col_32": Array(Float32),
    }

    @udf(
        params=(C("file.path"),),
        output=FILE_SCHEMA | new_columns,
        method="generate_subobjects",
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, path):
            from datachain.lib.file import File

            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                np.int32(11),
                np.int64(12),
                np.float32(0.5),
                np.float64(0.5),
                np.int32(13),
                np.array([[0.5], [0.5]], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float64),
                np.array([14, 15], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

    DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
        Subobjects
    ).save("dogs_with_rows_and_signals")

    q = DatasetQuery(name="dogs_with_rows_and_signals", catalog=catalog)
    result = q.to_db_records()

    col_values = [
        (
            r["file__path"],
            r["int_col_32"],
            r["int_col_64"],
            r["float_col_32"],
            r["float_col_64"],
            r["int_float_col_32"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
            r["array_int_float_col_32"],
            r["array_empty_col_32"],
        )
        for r in result
    ]
    col_values.sort(key=lambda x: x[0])

    new_col_values = (
        11,
        12,
        0.5,
        0.5,
        13.0,
        [[0.5], [0.5]],
        [0.5, 0.5],
        [0.5, 0.5],
        [14.0, 15.0],
        [],
    )

    assert col_values == [
        ("dogs/dog1/subobject", *new_col_values),
        ("dogs/dog2/subobject", *new_col_values),
        ("dogs/dog3/subobject", *new_col_values),
        ("dogs/others/dog4/subobject", *new_col_values),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_generator_with_new_columns_wrong_type(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(
        params=(C("file.path"),),
        output={**FILE_SCHEMA, "int_col": Int},
        method="generate_subobjects",
    )
    class Subobjects:
        def __init__(self):
            pass

        def generate_subobjects(self, path):
            from datachain.lib.file import File

            yield (
                *tuple(f[1] for f in list(File(path=f"{path}/subobject", size=50))),
                0.5,
            )

    with pytest.raises(ValueError):
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).generate(
            Subobjects
        ).to_db_records()


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
def test_index_tar(cloud_test_catalog, animal_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog

    q = DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog).generate(
        index_tar
    )
    q.save("extracted")

    assert_row_names(
        catalog,
        catalog.get_dataset("extracted"),
        1,
        {
            "animals.tar",
            "cat1",
            "cat2",
            "description",
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )

    rows = catalog.ls_dataset_rows("extracted", 1)

    offsets = [
        json.loads(row["file__location"])[0]["offset"]
        for row in rows
        if not row["file__path"].endswith("animals.tar")
    ]
    # Check that offsets are unique integers
    assert all(isinstance(offset, int) for offset in offsets)
    assert len(set(offsets)) == len(offsets)

    assert all(
        json.loads(row["file__location"])[0]["vtype"] == "tar"
        for row in rows
        if not row["file__path"].endswith("animals.tar")
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_checksum_udf(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog).add_signals(
        checksum
    )
    result = q.db_results()

    assert len(result) == 4


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
def test_tar_loader(cloud_test_catalog, animal_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    q = DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog).generate(
        index_tar
    )
    q.save("extracted")

    q = DatasetQuery(name="extracted", catalog=catalog).filter(
        C("file.path").glob("*/cats/*")
    )
    assert len(q.db_results()) == 2

    ds = q.extract(Object(to_str), "file__path")
    assert {(value, posixpath.basename(path)) for value, path in ds} == {
        ("meow", "cat1"),
        ("mrow", "cat2"),
    }


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
        dq = dr.select().order_by(dr.c.file__path)
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


def test_similarity_search(cloud_test_catalog):
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    catalog.create_dataset_from_sources("ds1", [ctc.src_uri], recursive=True)

    @udf(
        params=(Object(to_str),),
        output={"embedding": Array(Float32)},
    )
    def embedding(text):
        return (text_embedding(text),)

    (target_embedding,) = (
        DatasetQuery(name="ds1", catalog=catalog)
        .filter(C("file.path").glob("*description"))
        .order_by(sqlalchemy.func.length(C("file.path")))
        .limit(1)
        .add_signals(embedding)
        .select(C.embedding)
        .db_results()[0]
    )
    q = (
        DatasetQuery(name="ds1", catalog=catalog)
        .add_signals(embedding)
        .mutate(
            cos_dist=cosine_distance(C.embedding, target_embedding),
            eucl_dist=euclidean_distance(C.embedding, target_embedding),
        )
        .select(C("file.path"), C.cos_dist, C.eucl_dist)
        .order_by(C("file.path"))
    )
    count = q.count()
    assert count == 7

    result = q.db_results()
    expected = [
        ("cats/cat1", 0.8508677010357059, 1.9078358385397216),
        ("cats/cat2", 0.8508677010357059, 1.9078358385397216),
        ("description", 0.0, 0.0),
        ("dogs/dog1", 0.7875133863812602, 1.8750659656122843),
        ("dogs/dog2", 0.7356502722055684, 1.775619888314893),
        ("dogs/dog3", 0.7695916496857775, 1.8344983482620636),
        ("dogs/others/dog4", 0.9789704524691446, 2.0531542018152322),
    ]

    for (p1, c1, e1), (p2, c2, e2) in zip(result, expected):
        assert p1.endswith(p2)
        assert math.isclose(c1, c2, abs_tol=1e-5)
        assert math.isclose(e1, e2, abs_tol=1e-5)


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
def test_subtract(cloud_test_catalog, dogs_dataset, cats_dataset):
    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    catalog = cloud_test_catalog.catalog
    on = ["file__source", "file__path"]

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    (dogs | cats).save("dogs_cats")

    dogs_cats = DatasetQuery(name="dogs_cats", catalog=catalog)

    # subtracting dataset from dataset
    q = dogs_cats.subtract(dogs, on=on)
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == [
        "cat1",
        "cat2",
    ]

    # subtracting with filter
    q = (
        DatasetQuery(dogs_cats.name, catalog=catalog)
        .filter(C("file.path").glob("*dog*"))
        .subtract(cats, on=on)
    )
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == [
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    ]

    # chain subtracting
    q = dogs_cats.subtract(dogs, on=on).subtract(cats, on=on)
    result = q.db_results()
    count = q.count()
    assert len(result) == 0
    assert count == 0

    # filtering after subtract
    q = (
        DatasetQuery(dogs_cats.name, catalog=catalog)
        .subtract(cats, on=on)
        .filter(C("file.path").glob("*dog*"))
    )
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == [
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    ]

    # subtract with usage of udfs and union
    # simulates updating dataset with new changes in index and not re-calculating
    # all udfs, but only for those that are new
    cats.add_signals(name_len).save("cats_with_signals")
    cats_with_signals = DatasetQuery(name="cats_with_signals", catalog=catalog)
    q = (
        DatasetQuery(dogs_cats.name, catalog=catalog)
        .subtract(cats_with_signals, on=on)
        .add_signals(name_len)
        .union(cats_with_signals)
    )
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == sorted(
        ["dog1", "dog2", "dog3", "dog4", "cat1", "cat2"]
    )
    assert all(r["name_len"] > 0 for r in result)

    # subtracting with source and target filter
    # only dog2 file has size less then 4
    all_except_dog2 = DatasetQuery(dogs_cats.name, catalog=catalog).filter(
        C("file.size") > 3
    )
    only_cats = dogs_cats.filter(C("file.path").glob("*cat*"))
    q = all_except_dog2.subtract(only_cats, on=on)
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == [
        "dog1",
        "dog3",
        "dog4",
    ]

    # subtracting after union
    q = dogs.union(cats).subtract(dogs, on=on)
    result = q.db_results(row_factory=from_result_row)
    assert sorted(posixpath.basename(r["file__path"]) for r in result) == [
        "cat1",
        "cat2",
    ]

    # subtract with itself
    q = dogs.subtract(dogs, on=on)
    result = q.db_results()
    count = q.count()
    assert len(result) == 0
    assert count == 0


def test_aggregate(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    q = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    assert q.count() == 4
    assert q.sum(C("file.size")) == 15
    assert q.avg(C("file.size")) == 15 / 4
    assert q.min(C("file.size")) == 3
    assert q.max(C("file.size")) == 4


def test_group_by(cloud_test_catalog, cloud_type, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    q = (
        DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
        .mutate(parent=pathfunc.parent(C("file.path")))
        .group_by(C.parent)
        .select(
            C.parent,
            functions.count(),
            functions.sum(C("file.size")),
            functions.avg(C("file.size")),
            functions.min(C("file.size")),
            functions.max(C("file.size")),
        )
    )
    result = q.db_results()
    assert len(result) == 2

    result_dict = {r[0]: r[1:] for r in result}
    if cloud_type == "file":
        assert result_dict == {
            f"{cloud_test_catalog.partial_path}/dogs": (3, 11, 11 / 3, 3, 4),
            f"{cloud_test_catalog.partial_path}/dogs/others": (1, 4, 4, 4, 4),
        }

    else:
        assert result_dict == {
            "dogs": (3, 11, 11 / 3, 3, 4),
            "dogs/others": (1, 4, 4, 4, 4),
        }


@pytest.mark.parametrize("tree", [WEBFORMAT_TREE], indirect=True)
def test_json_loader(cloud_test_catalog, animal_dataset):
    ctc = cloud_test_catalog
    catalog = ctc.catalog

    @udf(
        params=(C.name,),
        output={"basename": String, "ext": String},
    )
    def split_name(name):
        return os.path.splitext(name)

    json_output = {"similarity": Float, "md5": String}

    @udf(
        params=("ext", LocalFilename("*.json")),
        output=json_output,
    )
    def attach_json(rows):
        # Locate json row and load its data
        json_data = None
        for ext, file_path in rows:
            if ext == ".json" and file_path:
                with open(file_path, encoding="utf8") as f:
                    json_data = json.load(f)

        # Attach json-loaded signals to all other rows in the group
        signals = []
        for ext, _ in rows:
            if json_data and ext != ".json":
                signals.append([json_data.get(k) for k in json_output])
            else:
                signals.append([None, None])

        return signals

    expected = [
        ("f1.raw", 0.001, "deadbeef"),
        ("f2.raw", 0.005, "foobar"),
    ]

    q = (
        DatasetQuery(animal_dataset.name, catalog=catalog)
        .mutate(name=pathfunc.name(C("file.path")))
        .add_signals(split_name)
        .add_signals(attach_json, partition_by=C.basename)
        .filter(C.glob(C.name, "*.raw"))
        .select(C.name, C.similarity, C.md5)
        .order_by(C.name)
    )
    assert q.count() == 2
    res = q.db_results()
    assert len(res) == 2
    assert [r[0] for r in res] == [r[0] for r in expected]
    assert [r[1] for r in res] == pytest.approx([r[1] for r in expected])
    assert [r[2] for r in res] == [r[2] for r in expected]


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


@pytest.mark.parametrize("method", ["to_db_records"])
@pytest.mark.parametrize("save", [True, False])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
def test_udf_after_union(cloud_test_catalog, save, method, animal_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    ds_cats = DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog).filter(
        C("file.path").glob("*cats*")
    )
    if save:
        ds_cats.save("cats")
        ds_cats = DatasetQuery(name="cats", version=1, catalog=catalog)
    ds_dogs = DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog).filter(
        C("file.path").glob("*dogs*")
    )
    if save:
        ds_dogs.save("dogs")
        ds_dogs = DatasetQuery(name="dogs", version=1, catalog=catalog)

    if method == "to_db_records":

        def get_result(query):
            result = [
                (posixpath.basename(r["file__path"]), r["name_len"])
                for r in query.to_db_records()
            ]
            result.sort()
            return result

    q = ds_cats.union(ds_dogs).add_signals(name_len)
    result1 = get_result(q)
    assert result1 == [
        ("cat1", 4),
        ("cat2", 4),
        ("dog1", 4),
        ("dog2", 4),
        ("dog3", 4),
        ("dog4", 4),
    ]

    result2 = get_result(q.union(q))
    assert result2 == [
        ("cat1", 4),
        ("cat1", 4),
        ("cat2", 4),
        ("cat2", 4),
        ("dog1", 4),
        ("dog1", 4),
        ("dog2", 4),
        ("dog2", 4),
        ("dog3", 4),
        ("dog3", 4),
        ("dog4", 4),
        ("dog4", 4),
    ]


@pytest.mark.parametrize("method", ["to_db_records"])
@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
def test_udf_after_union_same_rows_with_mutate(
    cloud_test_catalog, method, animal_dataset
):
    catalog = cloud_test_catalog.catalog

    @udf(("file__path",), {"name_len": Int})
    def name_len(path):
        return (len(posixpath.basename(path)),)

    q_base = DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog).filter(
        C("file.path").glob("*dogs*")
    )
    q1 = q_base.mutate(
        x=sqlalchemy.cast(pathfunc.name(C("file.path")) + "_1", String())
    )
    q2 = q_base.mutate(
        x=sqlalchemy.cast(pathfunc.name(C("file.path")) + "_2", String())
    )

    if method == "to_db_records":

        def get_result(query):
            result = [
                (posixpath.basename(r["file__path"]), r["x"], r["name_len"])
                for r in query.to_db_records()
            ]
            result.sort()
            return result

    q = q1.union(q2).add_signals(name_len)
    result1 = get_result(q)
    assert result1 == [
        ("dog1", "dog1_1", 4),
        ("dog1", "dog1_2", 4),
        ("dog2", "dog2_1", 4),
        ("dog2", "dog2_2", 4),
        ("dog3", "dog3_1", 4),
        ("dog3", "dog3_2", 4),
        ("dog4", "dog4_1", 4),
        ("dog4", "dog4_2", 4),
    ]

    result2 = get_result(q.union(q))
    assert result2 == [
        ("dog1", "dog1_1", 4),
        ("dog1", "dog1_1", 4),
        ("dog1", "dog1_2", 4),
        ("dog1", "dog1_2", 4),
        ("dog2", "dog2_1", 4),
        ("dog2", "dog2_1", 4),
        ("dog2", "dog2_2", 4),
        ("dog2", "dog2_2", 4),
        ("dog3", "dog3_1", 4),
        ("dog3", "dog3_1", 4),
        ("dog3", "dog3_2", 4),
        ("dog3", "dog3_2", 4),
        ("dog4", "dog4_1", 4),
        ("dog4", "dog4_1", 4),
        ("dog4", "dog4_2", 4),
        ("dog4", "dog4_2", 4),
    ]


@pytest.mark.parametrize("method", ["select"])
@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, NUM_TREE), ("file", False, NUM_TREE)],
    indirect=True,
)
def test_udf_after_limit(cloud_test_catalog, method, animal_dataset):
    catalog = cloud_test_catalog.catalog

    @udf(("name",), {"name_int": Int})
    def name_int(name):
        try:
            return (int(name),)
        except ValueError:
            return 0

    if method == "select":

        def get_result(query):
            return (
                query.limit(100)
                .add_signals(name_int)
                .select("name", "name_int")
                .to_db_records()
            )

    expected = [{"name": f"{i:06d}", "name_int": i} for i in range(100)]
    ds = (
        DatasetQuery(name=animal_dataset.name, version=1, catalog=catalog)
        .mutate(name=pathfunc.name(C("file.path")))
        .save()
    )
    # We test a few different orderings here, because we've had strange
    # bugs in the past where calling add_signals() after limit() gave us
    # incorrect results on clickhouse cloud.
    # See https://github.com/iterative/dvcx/issues/940
    assert get_result(ds.order_by(C.name)) == expected
    assert len(get_result(ds.order_by("sys__rand"))) == 100
    assert len(get_result(ds)) == 100


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True), ("file", False)],
    indirect=True,
)
@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_one_storage_as_dependency(
    cloud_test_catalog, listed_bucket, indirect, cats_dataset
):
    pytest.skip(
        "Skipping as new dependencies are not working with old indexing "
        "It will be fixed after https://github.com/iterative/datachain/issues/340"
    )
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    storage = catalog.metastore.get_storage(cloud_test_catalog.storage_uri)

    DatasetQuery(cats_dataset.name, catalog=catalog).save(ds_name)

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(ds_name, 1, indirect=indirect)
    ] == [
        {
            "id": ANY,
            "type": DatasetDependencyType.STORAGE,
            "name": storage.uri,
            "version": storage.timestamp_str,
            "created_at": isoparse(storage.timestamp_str),
            "dependencies": [],
        }
    ]


@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_one_registered_dataset_as_dependency(
    cloud_test_catalog, dogs_dataset, indirect
):
    pytest.skip(
        "Skipping as new dependencies are not working with old indexing "
        "It will be fixed after https://github.com/iterative/datachain/issues/340"
    )
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    storage = catalog.metastore.get_storage(cloud_test_catalog.storage_uri)

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
                "name": storage.uri,
                "version": storage.timestamp_str,
                "created_at": isoparse(storage.timestamp_str),
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
    pytest.skip(
        "Skipping as new dependencies are not working with old indexing "
        "It will be fixed after https://github.com/iterative/datachain/issues/340"
    )
    # multiple direct dataset dependencies can be achieved with methods that are
    # combining multiple DatasetQuery instances into new one like union or join
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    storage = catalog.metastore.get_storage(cloud_test_catalog.storage_uri)

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)

    if method == "union":
        dogs.union(cats).save(ds_name)
    else:
        dogs.join(cats, "file__path").save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": storage.uri,
        "version": storage.timestamp_str,
        "created_at": isoparse(storage.timestamp_str),
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
    pytest.skip(
        "Skipping as new dependencies are not working with old indexing "
        "It will be fixed after https://github.com/iterative/datachain/issues/340"
    )
    ds_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog
    storage = catalog.metastore.get_storage(cloud_test_catalog.storage_uri)

    dogs = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)
    cats = DatasetQuery(name=cats_dataset.name, version=1, catalog=catalog)
    dogs_other = DatasetQuery(name=dogs_dataset.name, version=1, catalog=catalog)

    dogs.union(cats).union(dogs_other).save(ds_name)

    storage_depenedncy = {
        "id": ANY,
        "type": DatasetDependencyType.STORAGE,
        "name": storage.uri,
        "version": storage.timestamp_str,
        "created_at": isoparse(storage.timestamp_str),
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
