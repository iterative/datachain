import posixpath
import uuid
from json import dumps
from unittest.mock import ANY

import pytest
import sqlalchemy as sa
from dateutil.parser import isoparse

from datachain.catalog.catalog import DATASET_INTERNAL_ERROR_MESSAGE
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.dataset import DatasetDependencyType, DatasetStatus
from datachain.error import DatasetInvalidVersionError, DatasetNotFoundError
from datachain.query import DatasetQuery, udf
from datachain.query.schema import DatasetRow
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    String,
)
from tests.utils import assert_row_names, dataset_dependency_asdict


def add_column(engine, table_name, column, catalog):
    # Simple method that adds new column to a table, with default value if specified
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    query_str = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
    if column.default:
        query_str += f" DEFAULT {column.default.arg}"
    catalog.warehouse.db.execute_str(query_str)


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_no_version_specified(cloud_test_catalog, create_rows):
    catalog = cloud_test_catalog.catalog

    name = uuid.uuid4().hex
    dataset = catalog.create_dataset(
        name,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    assert dataset.versions_values == [1]

    dataset_version = dataset.get_version(1)

    assert dataset.name == name
    assert dataset.query_script == "script"
    assert dataset_version.query_script == "script"
    assert dataset.schema["similarity"] == Float32
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.PENDING
    assert dataset.status == DatasetStatus.CREATED  # dataset status is deprecated
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_with_explicit_version(cloud_test_catalog, create_rows):
    catalog = cloud_test_catalog.catalog

    name = uuid.uuid4().hex
    dataset = catalog.create_dataset(
        name,
        version=1,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    assert dataset.versions_values == [1]

    dataset_version = dataset.get_version(1)

    assert dataset.name == name
    assert dataset.query_script == "script"
    assert dataset_version.query_script == "script"
    assert dataset.schema["similarity"] == Float32
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.PENDING
    assert dataset.status == DatasetStatus.CREATED
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_already_exist(cloud_test_catalog, dogs_dataset, create_rows):
    catalog = cloud_test_catalog.catalog

    dataset = catalog.create_dataset(
        dogs_dataset.name,
        query_script="script",
        columns=[sa.Column("similarity", Float32)],
        create_rows=create_rows,
    )

    assert dataset.versions_values == [1, 2]

    dataset_version = dataset.get_version(2)

    assert dataset.name == dogs_dataset.name
    assert dataset_version.query_script == "script"
    assert dataset_version.schema["similarity"] == Float32
    assert dataset_version.status == DatasetStatus.PENDING
    assert dataset.status == DatasetStatus.COMPLETE
    if create_rows:
        assert dataset_version.num_objects == 0
    else:
        assert dataset_version.num_objects is None


@pytest.mark.parametrize("create_rows", [True, False])
def test_create_dataset_already_exist_wrong_version(
    cloud_test_catalog, dogs_dataset, create_rows
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetInvalidVersionError) as exc_info:
        catalog.create_dataset(
            dogs_dataset.name,
            version=1,
            columns=[sa.Column(name, typ) for name, typ in dogs_dataset.schema.items()],
            create_rows=create_rows,
        )
    assert str(exc_info.value) == (
        f"Version 1 already exists in dataset {dogs_dataset.name}"
    )


def test_get_dataset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dataset = catalog.get_dataset(dogs_dataset.name)
    assert dataset.name == dogs_dataset.name

    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("wrong name")


# Returns None if the table does not exist
def get_table_row_count(db, table_name):
    if not db.has_table(table_name):
        return None
    query = sa.select(sa.func.count()).select_from(sa.table(table_name))
    return next(db.execute(query), (None,))[0]


def test_create_dataset_from_sources(listed_bucket, cloud_test_catalog):
    dataset_name = uuid.uuid4().hex
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    dataset = catalog.create_dataset_from_sources(
        dataset_name, [f"{src_uri}/dogs/*"], recursive=True
    )

    dataset_version = dataset.get_version(dataset.latest_version)

    assert dataset.name == dataset_name
    assert dataset.description is None
    assert dataset.versions_values == [1]
    assert dataset.labels == []
    assert dataset.status == DatasetStatus.COMPLETE

    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.error_message == ""
    assert dataset_version.error_stack == ""
    assert dataset_version.script_output == ""
    assert dataset_version.sources == f"{src_uri}/dogs/*"

    dr = catalog.warehouse.schema.dataset_row_cls
    sys_schema = {c.name: type(c.type) for c in dr.sys_columns()}
    default_dataset_schema = DatasetRow.schema | sys_schema
    assert dataset.schema == default_dataset_schema
    assert dataset.query_script == ""

    assert dataset_version.schema == default_dataset_schema
    assert dataset_version.query_script == ""
    assert dataset_version.num_objects
    assert dataset_version.preview


def test_create_dataset_from_sources_empty_sources(cloud_test_catalog):
    dataset_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    with pytest.raises(ValueError) as exc_info:
        catalog.create_dataset_from_sources(dataset_name, [], recursive=True)

    assert str(exc_info.value) == "Sources needs to be non empty list"


def test_create_dataset_from_sources_failed(listed_bucket, cloud_test_catalog, mocker):
    dataset_name = uuid.uuid4().hex
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog
    # Mocks are automatically undone at the end of a test.
    mocker.patch.object(
        catalog.warehouse.__class__,
        "create_dataset_rows_table",
        side_effect=RuntimeError("Error"),
    )
    with pytest.raises(RuntimeError):
        catalog.create_dataset_from_sources(
            dataset_name, [f"{src_uri}/dogs/*"], recursive=True
        )

    dataset = catalog.get_dataset(dataset_name)
    dataset_version = dataset.get_version(dataset.latest_version)

    assert dataset.name == dataset_name
    assert dataset.status == DatasetStatus.FAILED
    assert dataset.versions_values == [1]
    assert dataset.created_at
    assert dataset.finished_at
    assert dataset.error_message == DATASET_INTERNAL_ERROR_MESSAGE
    assert dataset.error_stack
    assert dataset.query_script == ""

    assert dataset_version.status == DatasetStatus.FAILED
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.error_message == DATASET_INTERNAL_ERROR_MESSAGE
    assert dataset_version.error_stack
    assert dataset_version.sources == f"{src_uri}/dogs/*"
    assert dataset_version.num_objects is None
    assert dataset_version.size is None
    assert dataset_version.preview is None


def test_create_dataset_whole_bucket(listed_bucket, cloud_test_catalog):
    dataset_name_1 = uuid.uuid4().hex
    dataset_name_2 = uuid.uuid4().hex
    src_uri = cloud_test_catalog.src_uri
    catalog = cloud_test_catalog.catalog

    ds1 = catalog.create_dataset_from_sources(
        dataset_name_1, [f"{src_uri}"], recursive=True
    )
    ds2 = catalog.create_dataset_from_sources(
        dataset_name_2, [f"{src_uri}/"], recursive=True
    )

    expected_rows = {
        "description",
        "cat1",
        "cat2",
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    }

    assert_row_names(catalog, ds1, ds1.latest_version, expected_rows)

    assert_row_names(catalog, ds2, ds2.latest_version, expected_rows)


@pytest.mark.parametrize("target_version", [None, 3])
def test_registering_dataset(
    cloud_test_catalog, dogs_dataset, cats_dataset, target_version
):
    catalog = cloud_test_catalog.catalog

    # make sure there is a custom columns inside, other than default ones
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        sources="s3://ldb-public",
        query_script="DatasetQuery()",
        error_message="no error",
        error_stack="no error stack",
        script_output="log",
    )

    dogs_version = dogs_dataset.get_version(1)

    dataset = catalog.register_dataset(
        dogs_dataset,
        1,
        cats_dataset,
        target_version=target_version,
    )

    # if not provided, it will end up being next dataset version
    target_version = target_version or cats_dataset.next_version

    assert dataset.name == cats_dataset.name
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.versions_values == [1, target_version]

    version1 = dataset.get_version(1)
    assert version1.status == DatasetStatus.COMPLETE

    version2 = dataset.get_version(target_version)
    assert version2.status == DatasetStatus.COMPLETE
    assert version2.sources == "s3://ldb-public"
    assert version2.query_script == "DatasetQuery()"
    assert version2.error_message == "no error"
    assert version2.error_stack == "no error stack"
    assert version2.script_output == "log"
    assert version2.schema == dogs_version.schema
    assert version2.created_at == dogs_version.created_at
    assert version2.finished_at == dogs_version.finished_at
    assert dogs_version.num_objects
    assert version2.num_objects == dogs_version.num_objects
    assert dogs_version.size
    assert version2.size == dogs_version.size

    assert_row_names(
        catalog,
        dataset,
        1,
        {
            "cat1",
            "cat2",
        },
    )

    assert_row_names(
        catalog,
        dataset,
        target_version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )

    with pytest.raises(DatasetNotFoundError):
        # since it had only one version, it should be completely removed
        catalog.get_dataset(dogs_dataset.name)


@pytest.mark.parametrize("target_version", [None, 3])
def test_registering_dataset_source_dataset_with_multiple_versions(
    cloud_test_catalog, dogs_dataset, cats_dataset, target_version
):
    catalog = cloud_test_catalog.catalog

    # creating one more version for dogs, not to end up completely removed
    columns = tuple(sa.Column(name, typ) for name, typ in dogs_dataset.schema.items())
    dogs_dataset = catalog.create_new_dataset_version(dogs_dataset, 2, columns=columns)
    dataset = catalog.register_dataset(
        dogs_dataset,
        1,
        cats_dataset,
        target_version=target_version,
    )

    # if not provided, it will end up being next dataset version
    target_version = target_version or cats_dataset.next_version

    assert dataset.name == cats_dataset.name
    assert dataset.versions_values == [1, target_version]

    assert_row_names(
        catalog,
        dataset,
        1,
        {
            "cat1",
            "cat2",
        },
    )

    assert_row_names(
        catalog,
        dataset,
        target_version,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )

    # check if dogs dataset is still present as it has one more version
    dogs_dataset = catalog.get_dataset(dogs_dataset.name)
    assert dogs_dataset.versions_values == [2]
    dataset_version = dogs_dataset.get_version(2)
    assert dataset_version.num_objects == 0


@pytest.mark.parametrize("target_version", [None, 3])
def test_registering_dataset_with_new_version_of_itself(
    cloud_test_catalog, cats_dataset, target_version
):
    catalog = cloud_test_catalog.catalog

    dataset = catalog.register_dataset(
        cats_dataset,
        1,
        cats_dataset,
        target_version=target_version,
    )

    # if not provided, it will end up being next dataset version
    target_version = target_version or cats_dataset.next_version

    assert dataset.name == cats_dataset.name
    assert dataset.versions_values == [target_version]

    assert_row_names(catalog, dataset, target_version, {"cat1", "cat2"})


def test_registering_dataset_invalid_target_version(
    cloud_test_catalog, cats_dataset, dogs_dataset
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetInvalidVersionError) as exc_info:
        catalog.register_dataset(
            dogs_dataset,
            1,
            cats_dataset,
            target_version=1,
        )
    assert str(exc_info.value) == "Version 1 must be higher than the current latest one"


def test_registering_dataset_invalid_source_version(
    cloud_test_catalog, cats_dataset, dogs_dataset
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(ValueError) as exc_info:
        catalog.register_dataset(
            dogs_dataset,
            5,
            cats_dataset,
            target_version=2,
        )
    assert str(exc_info.value) == f"Dataset {dogs_dataset.name} does not have version 5"


def test_registering_dataset_source_version_in_non_final_status(
    cloud_test_catalog, cats_dataset, dogs_dataset
):
    catalog = cloud_test_catalog.catalog
    catalog.metastore.update_dataset_version(
        dogs_dataset,
        1,
        status=DatasetStatus.PENDING,
    )

    with pytest.raises(ValueError) as exc_info:
        catalog.register_dataset(
            dogs_dataset,
            1,
            cats_dataset,
            target_version=2,
        )
    assert str(exc_info.value) == "Cannot register dataset version in non final status"


def test_remove_dataset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    dataset_version = dogs_dataset.get_version(1)
    assert dataset_version.num_objects

    catalog.remove_dataset(dogs_dataset.name, force=True)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(dogs_dataset.name)

    dataset_table_name = catalog.warehouse.dataset_table_name(dogs_dataset.name, 1)
    assert get_table_row_count(catalog.warehouse.db, dataset_table_name) is None

    assert catalog.metastore.get_direct_dataset_dependencies(dogs_dataset, 1) == []


def test_remove_dataset_with_multiple_versions(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    columns = tuple(sa.Column(name, typ) for name, typ in dogs_dataset.schema.items())
    updated_dogs_dataset = catalog.create_new_dataset_version(
        dogs_dataset, 2, columns=columns
    )
    assert updated_dogs_dataset.has_version(2)
    assert updated_dogs_dataset.has_version(1)

    catalog.remove_dataset(updated_dogs_dataset.name, force=True)
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset(updated_dogs_dataset.name)

    assert (
        catalog.metastore.get_direct_dataset_dependencies(updated_dogs_dataset, 1) == []
    )


def test_remove_dataset_dataset_not_found(cloud_test_catalog):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetNotFoundError):
        catalog.remove_dataset("wrong_name", force=True)


def test_remove_dataset_wrong_version(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DatasetInvalidVersionError):
        catalog.remove_dataset(dogs_dataset.name, version=100)


def test_edit_dataset(cloud_test_catalog, dogs_dataset):
    dataset_old_name = dogs_dataset.name
    dataset_new_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(
        dogs_dataset.name,
        new_name=dataset_new_name,
        description="new description",
        labels=["cats", "birds"],
    )

    dataset = catalog.get_dataset(dataset_new_name)
    assert dataset.versions_values == [1]
    assert dataset.name == dataset_new_name
    assert dataset.description == "new description"
    assert dataset.labels == ["cats", "birds"]

    # check if dataset tables are renamed correctly
    old_dataset_table_name = catalog.warehouse.dataset_table_name(dataset_old_name, 1)
    new_dataset_table_name = catalog.warehouse.dataset_table_name(dataset_new_name, 1)
    assert get_table_row_count(catalog.warehouse.db, old_dataset_table_name) is None
    expected_table_row_count = get_table_row_count(
        catalog.warehouse.db, new_dataset_table_name
    )
    assert expected_table_row_count
    assert dataset.get_version(1).num_objects == expected_table_row_count


def test_edit_dataset_same_name(cloud_test_catalog, dogs_dataset):
    dataset_old_name = dogs_dataset.name
    dataset_new_name = dogs_dataset.name
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(dogs_dataset.name, new_name=dataset_new_name)

    dataset = catalog.get_dataset(dataset_new_name)
    assert dataset.name == dataset_new_name

    # check if dataset tables are renamed correctly
    old_dataset_table_name = catalog.warehouse.dataset_table_name(dataset_old_name, 1)
    new_dataset_table_name = catalog.warehouse.dataset_table_name(dataset_new_name, 1)
    expected_table_row_count = get_table_row_count(
        catalog.warehouse.db, old_dataset_table_name
    )
    assert expected_table_row_count
    assert dataset.get_version(1).num_objects == expected_table_row_count
    assert expected_table_row_count == get_table_row_count(
        catalog.warehouse.db, new_dataset_table_name
    )


def test_edit_dataset_remove_labels_and_description(cloud_test_catalog, dogs_dataset):
    dataset_new_name = uuid.uuid4().hex
    catalog = cloud_test_catalog.catalog

    catalog.edit_dataset(
        dogs_dataset.name,
        new_name=dataset_new_name,
        description="",
        labels=[],
    )

    dataset = catalog.get_dataset(dataset_new_name)
    assert dataset.versions_values == [1]
    assert dataset.name == dataset_new_name
    assert dataset.description == ""
    assert dataset.labels == []


def test_ls_dataset_rows(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    assert {
        posixpath.basename(r["path"])
        for r in catalog.ls_dataset_rows(dogs_dataset.name, 1)
    } == {
        "dog1",
        "dog2",
        "dog3",
        "dog4",
    }


def test_ls_dataset_rows_with_limit_offset(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    # these should be sorted by id already
    all_rows = list(
        catalog.ls_dataset_rows(
            dogs_dataset.name,
            1,
        )
    )

    assert {
        r["path"]
        for r in catalog.ls_dataset_rows(
            dogs_dataset.name,
            1,
            offset=2,
            limit=1,
        )
    } == {
        all_rows[2]["path"],
    }


def test_ls_dataset_rows_with_custom_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    int_example = 25

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
            int_example.to_bytes(2, "big"),
        )

    (
        DatasetQuery(name=dogs_dataset.name, catalog=catalog)
        .add_signals(test_types)
        .save("dogs_custom_columns")
    )

    for r in catalog.ls_dataset_rows("dogs_custom_columns", 1):
        assert r["int_col"] == 5
        assert r["int_col_32"] == 5
        assert r["int_col_64"] == 5
        assert r["float_col"] == 0.5
        assert r["float_col_32"] == 0.5
        assert r["float_col_64"] == 0.5
        assert r["array_col"] == [0.5]
        assert r["array_col_nested"] == [[0.5], [0.5]]
        assert r["array_col_32"] == [0.5]
        assert r["array_col_64"] == [0.5]
        assert r["string_col"] == "s"
        assert r["bool_col"]
        assert r["json_col"] == dumps({"a": 1})
        assert r["binary_col"] == int_example.to_bytes(2, "big")


def test_dataset_preview_custom_columns(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    int_example = 25

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
            int_example.to_bytes(2, "big"),
        )

    (
        DatasetQuery(name=dogs_dataset.name, catalog=catalog)
        .add_signals(test_types)
        .save("dogs_custom_columns")
    )

    for r in catalog.get_dataset("dogs_custom_columns").get_version(1).preview:
        assert r["int_col"] == 5
        assert r["int_col_32"] == 5
        assert r["int_col_64"] == 5
        assert r["float_col"] == 0.5
        assert r["float_col_32"] == 0.5
        assert r["float_col_64"] == 0.5
        assert r["array_col"] == [0.5]
        assert r["array_col_nested"] == [[0.5], [0.5]]
        assert r["array_col_32"] == [0.5]
        assert r["array_col_64"] == [0.5]
        assert r["string_col"] == "s"
        assert r["bool_col"]
        assert r["json_col"] == '{"a": 1}'
        assert r["binary_col"] == [0, 25]


def test_dataset_preview_last_modified(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog

    DatasetQuery(name=dogs_dataset.name, catalog=catalog).save("dogs_custom_columns")

    for r in catalog.get_dataset("dogs_custom_columns").get_version(1).preview:
        assert isinstance(r.get("last_modified"), str)


@pytest.mark.parametrize("tree", [{str(i): str(i) for i in range(50)}], indirect=True)
def test_row_random(cloud_test_catalog):
    # Note: this is technically a probabilistic test, but the probability
    # of accidental failure is < 1e-10
    ctc = cloud_test_catalog
    catalog = ctc.catalog
    catalog.index([ctc.src_uri])
    catalog.create_dataset_from_sources("test", [ctc.src_uri])
    random_values = [row["sys__rand"] for row in catalog.ls_dataset_rows("test", 1)]

    # Random values are unique
    assert len(set(random_values)) == len(random_values)

    if isinstance(catalog.warehouse, SQLiteWarehouse):
        RAND_MAX = 2**63  # noqa: N806
    else:
        RAND_MAX = 2**64  # noqa: N806

    # Values are drawn uniformly from range(2**63)
    assert 0 <= min(random_values) < 0.4 * RAND_MAX
    assert 0.6 * RAND_MAX < max(random_values) < RAND_MAX

    # Creating a new dataset preserves random values
    catalog.create_dataset_from_sources("test2", [ctc.src_uri])
    random_values2 = {row["sys__rand"] for row in catalog.ls_dataset_rows("test2", 1)}
    assert random_values2 == set(random_values)


def test_dataset_stats_registered_ds(cloud_test_catalog, dogs_dataset):
    catalog = cloud_test_catalog.catalog
    stats = catalog.dataset_stats(dogs_dataset.name, 1)
    assert stats.num_objects == 4
    assert stats.size == 15
    rows_count = catalog.warehouse.dataset_rows_count(dogs_dataset, 1)
    assert rows_count == 4


@pytest.mark.parametrize("indirect", [True, False])
def test_dataset_dependencies_registered(
    listed_bucket, cloud_test_catalog, dogs_dataset, indirect
):
    catalog = cloud_test_catalog.catalog
    storage = catalog.get_storage(cloud_test_catalog.storage_uri)

    assert [
        dataset_dependency_asdict(d)
        for d in catalog.get_dataset_dependencies(
            dogs_dataset.name, 1, indirect=indirect
        )
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
