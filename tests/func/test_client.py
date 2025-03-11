import asyncio
import os
import sys
from pathlib import Path

import pytest
from fsspec.asyn import sync
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from tqdm import tqdm

from datachain.asyn import get_loop
from datachain.client import Client
from tests.data import ENTRIES

_non_null_text = st.text(
    alphabet=st.characters(blacklist_categories=["Cc", "Cs"]), min_size=1
)


@pytest.fixture
def client(cloud_server, cloud_server_credentials):
    uri = cloud_server.src_uri
    return Client.get_implementation(uri).from_source(
        uri, cache=None, **cloud_server.client_config
    )


def normalize_entries(entries):
    return {e.path for e in entries}


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


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rel_path=_non_null_text)
def test_parse_url(cloud_test_catalog, rel_path, cloud_type):
    if cloud_type == "file":
        assume(not rel_path.startswith("/"))
    bucket_uri = cloud_test_catalog.src_uri
    url = f"{bucket_uri}/{rel_path}"
    uri, rel_part = Client.parse_url(url)
    if cloud_type == "file":
        assert uri == url.rsplit("/", 1)[0]
        assert rel_part == url.rsplit("/", 1)[1]
    else:
        assert uri == bucket_uri
        assert rel_part == rel_path


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rel_path=_non_null_text)
def test_get_client(cloud_test_catalog, rel_path, cloud_type):
    catalog = cloud_test_catalog.catalog
    bucket_uri = cloud_test_catalog.src_uri
    url = f"{bucket_uri}/{rel_path}"
    client = Client.get_client(url, catalog.cache)
    assert client
    assert client.uri


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_absolute_path_without_protocol(cloud_test_catalog):
    working_dir = Path().absolute()
    uri, rel_part = Client.parse_url(str(working_dir / Path("animals")))
    assert uri == working_dir.as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_relative_path_multiple_dirs_back(cloud_test_catalog):
    uri, rel_part = Client.parse_url("../../animals".replace("/", os.sep))
    assert uri == Path().absolute().parents[1].as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
@pytest.mark.parametrize("url", ["./animals".replace("/", os.sep), "animals"])
def test_parse_file_relative_path_working_dir(cloud_test_catalog, url):
    uri, rel_part = Client.parse_url(url)
    assert uri == Path().absolute().as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_relative_path_home_dir(cloud_test_catalog):
    if sys.platform == "win32":
        # home dir shortcut is not available on windows
        pytest.skip()
    uri, rel_part = Client.parse_url("~/animals")
    assert uri == Path().home().as_uri()
    assert rel_part == "animals"


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_parse_cloud_path_ends_with_slash(cloud_test_catalog):
    uri = f"{cloud_test_catalog.src_uri}/animals/"
    uri, rel_part = Client.parse_url(uri)
    assert uri == cloud_test_catalog.src_uri
    assert rel_part == "animals/"
