import io
import json
from datetime import datetime

import lz4.frame
import pandas as pd
import pytest

import datachain as dc
from datachain.client.fsspec import Client
from datachain.dataset import DatasetStatus
from datachain.error import DataChainError, DatasetNotFoundError
from datachain.query.session import Session
from datachain.utils import STUDIO_URL, JSONSerialize
from tests.data import ENTRIES
from tests.utils import assert_row_names, skip_if_not_sqlite, tree_from_path

DATASET_UUID = "20f5a2f1-fc9a-4e36-8b91-5a530f289451"


@pytest.fixture
def dog_entries():
    # TODO remove when we replace ENTRIES with FILES
    return [
        {
            "file__path": e.path,
            "file__etag": e.etag,
            "file__version": e.version,
            "file__is_latest": e.is_latest,
            "file__last_modified": e.last_modified,
            "file__size": e.size,
        }
        for e in ENTRIES
        if e.name.startswith("dog")
    ]


@pytest.fixture
def dog_entries_parquet_lz4(dog_entries, cloud_test_catalog) -> bytes:
    """
    Returns dogs entries in lz4 compressed parquet format
    """
    src_uri = cloud_test_catalog.src_uri

    def _adapt_row(row):
        """
        Adjusting row values to match remote response
        """
        adapted = {}
        for k, v in row.items():
            if isinstance(v, datetime):
                adapted[k] = v.timestamp()
            elif v is None:
                adapted[k] = ""
            else:
                adapted[k] = v

        adapted["sys__id"] = 1
        adapted["sys__rand"] = 1
        adapted["file__location"] = ""
        adapted["file__source"] = src_uri
        adapted["file__version"] = ""
        return adapted

    dog_entries = [_adapt_row(e) for e in dog_entries]
    df = pd.DataFrame.from_records(dog_entries)
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="auto")

    return lz4.frame.compress(buffer.getvalue())


@pytest.fixture
def schema():
    return {
        "id": {"type": "UInt64"},
        "sys__rand": {"type": "UInt64"},
        "file__path": {"type": "String"},
        "file__etag": {"type": "String"},
        "file__version": {"type": "String"},
        "file__is_latest": {"type": "Boolean"},
        "file__last_modified": {"type": "DateTime"},
        "file__size": {"type": "Int64"},
        "file__location": {"type": "String"},
        "file__source": {"type": "String"},
    }


@pytest.fixture
def remote_dataset_version(schema, dataset_rows):
    return {
        "id": 1,
        "uuid": DATASET_UUID,
        "dataset_id": 1,
        "version": "1.0.0",
        "status": 4,
        "feature_schema": {},
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "finished_at": "2024-02-23T10:43:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "num_objects": 5,
        "size": 1073741824,
        "preview": json.loads(json.dumps(dataset_rows, cls=JSONSerialize)),
        "script_output": "",
        "schema": schema,
        "sources": "",
        "query_script": (
            'from datachain.query.dataset import DatasetQuery\nDatasetQuery(path="s3://ldb-public")',
        ),
        "created_by_id": 1,
    }


@pytest.fixture
def remote_dataset(remote_dataset_version, schema):
    return {
        "id": 1,
        "name": "dogs",
        "description": "",
        "attrs": [],
        "schema": schema,
        "status": 4,
        "feature_schema": {},
        "created_at": "2024-02-23T10:42:31.842944+00:00",
        "finished_at": "2024-02-23T10:43:31.842944+00:00",
        "error_message": "",
        "error_stack": "",
        "script_output": "",
        "job_id": "f74ec414-58b7-437d-81c5-d41e5365abba",
        "sources": "",
        "query_script": "",
        "team_id": 1,
        "warehouse_id": None,
        "created_by_id": 1,
        "versions": [remote_dataset_version],
    }


@pytest.fixture
def remote_dataset_chunk_url():
    return (
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0.parquet.lz4"
    )


@pytest.fixture
def remote_dataset_info(requests_mock, remote_dataset):
    requests_mock.get(f"{STUDIO_URL}/api/datachain/datasets/info", json=remote_dataset)


@pytest.fixture
def dataset_export(requests_mock, remote_dataset_chunk_url):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export", json=[remote_dataset_chunk_url]
    )


@pytest.fixture
def dataset_export_status(requests_mock):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={"status": "completed"},
    )


@pytest.fixture
def dataset_export_data_chunk(
    requests_mock, remote_dataset_chunk_url, dog_entries_parquet_lz4
):
    requests_mock.get(remote_dataset_chunk_url, content=dog_entries_parquet_lz4)


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("dataset_uri", ["ds://dogs@v1.0.0", "ds://dogs"])
@pytest.mark.parametrize("local_ds_name", [None, "other"])
@pytest.mark.parametrize("local_ds_version", [None, "2.0.0"])
@pytest.mark.parametrize("instantiate", [True, False])
@skip_if_not_sqlite
def test_pull_dataset_success(
    mocker,
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
    dataset_uri,
    local_ds_name,
    local_ds_version,
    instantiate,
):
    mocker.patch(
        "datachain.catalog.catalog.DatasetRowsFetcher.should_check_for_status",
        return_value=True,
    )

    src_uri = cloud_test_catalog.src_uri
    working_dir = cloud_test_catalog.working_dir
    catalog = cloud_test_catalog.catalog

    dest = None

    if instantiate:
        dest = working_dir / "data"
        dest.mkdir()
        catalog.pull_dataset(
            dataset_uri,
            output=str(dest),
            local_ds_name=local_ds_name,
            local_ds_version=local_ds_version,
            cp=True,
        )
    else:
        # trying to pull multiple times since that should work as well
        for _ in range(2):
            catalog.pull_dataset(
                dataset_uri,
                local_ds_name=local_ds_name,
                local_ds_version=local_ds_version,
                cp=False,
            )

    dataset = catalog.get_dataset(local_ds_name or "dogs")
    assert [v.version for v in dataset.versions] == [local_ds_version or "1.0.0"]
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.created_at
    assert dataset.finished_at
    assert dataset.schema
    dataset_version = dataset.get_version(local_ds_version or "1.0.0")
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.schema
    assert dataset_version.num_objects == 4
    assert dataset_version.size == 15
    assert dataset_version.uuid == DATASET_UUID

    assert_row_names(
        catalog,
        dataset,
        local_ds_version or "1.0.0",
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )

    client = Client.get_client(src_uri, None)

    if instantiate:
        assert tree_from_path(dest) == {
            f"{client.name}": {
                "dogs": {
                    "dog1": "woof",
                    "dog2": "arf",
                    "dog3": "bark",
                    "others": {"dog4": "ruff"},
                }
            }
        }


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_datachain_read_dataset_pull(
    mocker,
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
):
    # Check if the datachain pull from studio if datachain is not available.
    mocker.patch(
        "datachain.catalog.catalog.DatasetRowsFetcher.should_check_for_status",
        return_value=True,
    )

    catalog = cloud_test_catalog.catalog

    # Makes sure dataset is not available locally at first
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("dogs")

    with Session("testSession", catalog=catalog):
        ds = dc.read_dataset(
            name="dogs",
            version="1.0.0",
            fallback_to_studio=True,
        )

    assert ds.dataset.name == "dogs"
    assert ds.dataset.latest_version == "1.0.0"
    assert ds.dataset.status == DatasetStatus.COMPLETE

    # Check that dataset is available locally after pulling
    dataset = catalog.get_dataset("dogs")
    assert dataset.name == "dogs"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_wrong_dataset_uri_format(
    studio_token,
    requests_mock,
    cloud_test_catalog,
    remote_dataset,
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("wrong")
    assert str(exc_info.value) == "Error when parsing dataset uri"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_wrong_version(
    studio_token,
    requests_mock,
    cloud_test_catalog,
    remote_dataset_info,
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v5")
    assert str(exc_info.value) == "Dataset dogs doesn't have version 5 on server"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_not_found_in_remote(
    studio_token,
    requests_mock,
    cloud_test_catalog,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/info",
        status_code=404,
        json={"message": "Dataset not found"},
    )
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1.0.0")
    assert str(exc_info.value) == "Dataset not found"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("export_status", ["failed", "removed"])
@skip_if_not_sqlite
def test_pull_dataset_exporting_dataset_failed_in_remote(
    studio_token,
    requests_mock,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    export_status,
):
    requests_mock.get(
        f"{STUDIO_URL}/api/datachain/datasets/export-status",
        json={"status": export_status},
    )

    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1.0.0")
    assert str(exc_info.value) == f"Dataset export {export_status} in Studio"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_empty_parquet(
    studio_token,
    requests_mock,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    remote_dataset_chunk_url,
):
    requests_mock.get(remote_dataset_chunk_url, content=b"")
    catalog = cloud_test_catalog.catalog

    with pytest.raises(RuntimeError):
        catalog.pull_dataset("ds://dogs@v1.0.0")


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_already_exists_locally(
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
):
    catalog = cloud_test_catalog.catalog

    catalog.pull_dataset("ds://dogs@v1.0.0", local_ds_name="other")
    catalog.pull_dataset("ds://dogs@v1.0.0")

    other = catalog.get_dataset("other")
    other_version = other.get_version("1.0.0")
    assert other_version.uuid == DATASET_UUID
    assert other_version.num_objects == 4
    assert other_version.size == 15

    # dataset with same uuid created only once, on first pull with local name "other"
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("dogs")


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("local_ds_name", [None, "other"])
@skip_if_not_sqlite
def test_pull_dataset_local_name_already_exists(
    studio_token,
    cloud_test_catalog,
    remote_dataset_info,
    dataset_export,
    dataset_export_status,
    dataset_export_data_chunk,
    local_ds_name,
):
    catalog = cloud_test_catalog.catalog
    src_uri = cloud_test_catalog.src_uri

    catalog.create_dataset_from_sources(
        local_ds_name or "dogs", [f"{src_uri}/dogs/*"], recursive=True
    )
    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1.0.0", local_ds_name=local_ds_name)

    assert str(exc_info.value) == (
        f"Local dataset ds://{local_ds_name or 'dogs'}@v1.0.0 already exists with"
        " different uuid, please choose different local dataset name or version"
    )

    # able to save it as version 2 of local dataset name
    catalog.pull_dataset(
        "ds://dogs@v1.0.0", local_ds_name=local_ds_name, local_ds_version="2.0.0"
    )
