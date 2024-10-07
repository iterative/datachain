import io
import json
from datetime import datetime

import lz4.frame
import pandas as pd
import pytest

from datachain.dataset import DatasetStatus
from datachain.error import DataChainError
from datachain.utils import JSONSerialize
from tests.data import ENTRIES
from tests.utils import assert_row_names, skip_if_not_sqlite


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
def dog_entries_parquet_lz4(dog_entries) -> bytes:
    """
    Returns dogs entries in lz4 compressed parquet format
    """

    def _adapt_row(row):
        """
        Adjusting row values to match remote response
        """
        adapted = {}
        for k, v in row.items():
            if isinstance(v, str):
                adapted[k] = v.encode("utf-8")
            elif isinstance(v, datetime):
                adapted[k] = v.timestamp()
            elif v is None:
                adapted[k] = b""
            else:
                adapted[k] = v

        adapted["sys__id"] = 1
        adapted["sys__rand"] = 1
        adapted["file__location"] = b""
        adapted["file__source"] = b"s3://dogs"
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
        "sys__rand": {"type": "Int64"},
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
        "dataset_id": 1,
        "version": 1,
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
            'from datachain.query import DatasetQuery\nDatasetQuery(path="s3://ldb-public")',
        ),
        "created_by_id": 1,
    }


@pytest.fixture
def remote_dataset(remote_dataset_version, schema):
    return {
        "id": 1,
        "name": "remote",
        "description": "",
        "labels": [],
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
def remote_config():
    return {
        "url": "http://localhost:8111/api/datachain",
        "username": "datachain",
        "token": "isat_1LZKasZwyM46eHk6NHZZh4VCbHPRKUlQaLnYUE1bXb2U8Il0U",
    }


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("dataset_uri", ["ds://dogs@v1", "ds://dogs"])
@skip_if_not_sqlite
def test_pull_dataset_success(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
    dataset_uri,
):
    data_url = (
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0.parquet.lz4"
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-info', json=remote_dataset)
    requests_mock.post(
        f'{remote_config["url"]}/dataset-stats',
        json={"num_objects": 5, "size": 1000},
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-export', json=[data_url])
    requests_mock.post(
        f'{remote_config["url"]}/dataset-export-status',
        json={"status": "completed"},
    )
    requests_mock.get(data_url, content=dog_entries_parquet_lz4)
    catalog = cloud_test_catalog.catalog

    catalog.pull_dataset(dataset_uri, no_cp=True, remote_config=remote_config)
    # trying to pull multiple times as it should work
    catalog.pull_dataset(dataset_uri, no_cp=True, remote_config=remote_config)

    dataset = catalog.get_dataset("dogs")
    assert dataset.versions_values == [1]
    assert dataset.status == DatasetStatus.COMPLETE
    assert dataset.created_at
    assert dataset.finished_at
    assert dataset.schema
    dataset_version = dataset.get_version(1)
    assert dataset_version.status == DatasetStatus.COMPLETE
    assert dataset_version.created_at
    assert dataset_version.finished_at
    assert dataset_version.schema
    assert dataset_version.num_objects == 4
    assert dataset_version.size == 15

    assert_row_names(
        catalog,
        dataset,
        1,
        {
            "dog1",
            "dog2",
            "dog3",
            "dog4",
        },
    )


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_wrong_dataset_uri_format(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
):
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("wrong", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Error when parsing dataset uri"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_wrong_version(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    requests_mock.post(
        f'{remote_config["url"]}/dataset-info',
        json=remote_dataset,
    )
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v5", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Dataset dogs doesn't have version 5 on server"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_not_found_in_remote(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    requests_mock.post(
        f'{remote_config["url"]}/dataset-info',
        status_code=404,
        json={"message": "Dataset not found"},
    )
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Error from server: Dataset not found"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_error_on_fetching_stats(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    requests_mock.post(
        f'{remote_config["url"]}/dataset-info',
        json=remote_dataset,
    )
    requests_mock.post(
        f'{remote_config["url"]}/dataset-stats',
        status_code=400,
        json={"message": "Internal error"},
    )
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Error from server: Internal error"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@pytest.mark.parametrize("export_status", ["failed", "removed"])
@skip_if_not_sqlite
def test_pull_dataset_exporting_dataset_failed_in_remote(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    export_status,
):
    data_url = (
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0.parquet.lz4"
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-info', json=remote_dataset)
    requests_mock.post(
        f'{remote_config["url"]}/dataset-stats',
        json={"num_objects": 5, "size": 1000},
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-export', json=[data_url])
    requests_mock.post(
        f'{remote_config["url"]}/dataset-export-status',
        json={"status": export_status},
    )

    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v1", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == (
        f"Error from server: Dataset export {export_status} in Studio"
    )


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
@skip_if_not_sqlite
def test_pull_dataset_empty_parquet(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
):
    data_url = (
        "https://studio-blobvault.s3.amazonaws.com/datachain_ds_export_1_0.parquet.lz4"
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-info', json=remote_dataset)
    requests_mock.post(
        f'{remote_config["url"]}/dataset-stats',
        json={"num_objects": 5, "size": 1000},
    )
    requests_mock.post(f'{remote_config["url"]}/dataset-export', json=[data_url])
    requests_mock.post(
        f'{remote_config["url"]}/dataset-export-status',
        json={"status": "completed"},
    )
    requests_mock.get(data_url, content=b"")
    catalog = cloud_test_catalog.catalog

    with pytest.raises(RuntimeError):
        catalog.pull_dataset("ds://dogs@v1", no_cp=True, remote_config=remote_config)
