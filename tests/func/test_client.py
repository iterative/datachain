import asyncio

import pytest
from fsspec.asyn import sync
from tqdm import tqdm

from datachain.asyn import get_loop
from datachain.client import Client
from tests.data import ENTRIES


@pytest.fixture
def client(cloud_server, cloud_server_credentials):
    uri = cloud_server.src_uri
    return Client.get_implementation(uri).from_source(
        uri, cache=None, **cloud_server.client_config
    )


def normalize_entries(entries):
    return {(e.parent, e.name) for e in entries}


def match_entries(result, expected):
    assert len(result) == len(expected)
    assert normalize_entries(result) == normalize_entries(expected)


async def find(client, prefix, method="default"):
    results = []
    async for entries in client.scandir(prefix, method=method):
        results.extend(entries)
    return results


def scandir(client, prefix, method="default"):
    return sync(get_loop(), find, client, prefix, method)


def test_scandir_error(client):
    with pytest.raises(FileNotFoundError):
        scandir(client, "bogus")


@pytest.mark.xfail
def test_scandir_not_dir(client):
    with pytest.raises(FileNotFoundError):
        scandir(client, "description")


def test_scandir_success(client):
    results = scandir(client, "")
    match_entries(results, ENTRIES)


@pytest.mark.parametrize("cloud_type", ["s3", "gs", "azure"], indirect=True)
def test_scandir_alternate(client):
    results = scandir(client, "", method="nested")
    match_entries(results, ENTRIES)


def test_gcs_client_gets_credentials_from_env(monkeypatch, mocker):
    from datachain.client.gcs import GCSClient

    monkeypatch.setenv(
        "DATACHAIN_GCP_CREDENTIALS", '{"token": "test-credentials-token"}'
    )
    init = mocker.patch(
        "datachain.client.gcs.GCSFileSystem.__init__", return_value=None
    )
    mocker.patch(
        "datachain.client.gcs.GCSFileSystem.invalidate_cache", return_value=None
    )

    GCSClient.create_fs()

    init.assert_called_once_with(
        token={"token": "test-credentials-token"}, version_aware=True
    )


@pytest.mark.parametrize("tree", [{}], indirect=True)
def test_fetch_dir_does_not_return_self(client, cloud_type):
    if cloud_type == "file":
        pytest.skip()

    client.fs.touch(f"{client.uri}/directory//file")

    subdirs = sync(
        get_loop(), client._fetch_dir, "directory/", tqdm(disable=True), asyncio.Queue()
    )

    assert "directory" not in subdirs
