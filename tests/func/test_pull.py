import io
import json
from datetime import datetime

import attrs
import lz4.frame
import pandas as pd
import pytest

from datachain.dataset import DatasetStatus
from datachain.error import DataChainError
from datachain.node import DirType
from datachain.utils import JSONSerialize
from tests.data import ENTRIES
from tests.utils import assert_row_names, skip_if_not_sqlite


@pytest.fixture
def dog_entries():
    return [
        attrs.asdict(e) for e in ENTRIES if e.name.startswith("dog") and not e.is_dir
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

        adapted["id"] = 1
        adapted["vtype"] = b""
        adapted["location"] = b""
        adapted["source"] = b"s3://dogs"
        adapted["dir_type"] = DirType.FILE
        adapted["random"] = 1
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
        "vtype": {"type": "String"},
        "dir_type": {"type": "Int32"},
        "parent": {"type": "String"},
        "name": {"type": "String"},
        "etag": {"type": "String"},
        "version": {"type": "String"},
        "is_latest": {"type": "Boolean"},
        "last_modified": {"type": "DateTime"},
        "size": {"type": "Int64"},
        "owner_name": {"type": "String"},
        "owner_id": {"type": "String"},
        "random": {"type": "Int64"},
        "location": {"type": "String"},
        "source": {"type": "String"},
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
        "shadow": False,
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
def test_pull_dataset_success(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
    dataset_uri,
):
    skip_if_not_sqlite()
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
def test_pull_dataset_wrong_dataset_uri_format(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
):
    skip_if_not_sqlite()
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("wrong", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Error when parsing dataset uri"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
def test_pull_dataset_wrong_version(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    skip_if_not_sqlite()
    requests_mock.post(
        f'{remote_config["url"]}/dataset-info',
        json=remote_dataset,
    )
    catalog = cloud_test_catalog.catalog

    with pytest.raises(DataChainError) as exc_info:
        catalog.pull_dataset("ds://dogs@v5", no_cp=True, remote_config=remote_config)
    assert str(exc_info.value) == "Dataset dogs doesn't have version 5 on server"


@pytest.mark.parametrize("cloud_type, version_aware", [("s3", False)], indirect=True)
def test_pull_dataset_not_found_in_remote(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    skip_if_not_sqlite()
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
def test_pull_dataset_error_on_fetching_stats(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
):
    skip_if_not_sqlite()
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
def test_pull_dataset_exporting_dataset_failed_in_remote(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    export_status,
):
    skip_if_not_sqlite()
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
def test_pull_dataset_empty_parquet(
    requests_mock,
    cloud_test_catalog,
    remote_config,
    remote_dataset,
    dog_entries_parquet_lz4,
):
    skip_if_not_sqlite()
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
