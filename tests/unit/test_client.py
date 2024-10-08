import os
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from datachain.client import Client
from datachain.client.local import FileClient
from tests.utils import uppercase_scheme

non_null_text = st.text(
    alphabet=st.characters(blacklist_categories=["Cc", "Cs"]), min_size=1
)


def test_bad_protocol():
    with pytest.raises(NotImplementedError):
        Client.get_implementation("bogus://bucket")


def test_win_paths_are_recognized():
    if sys.platform != "win32":
        pytest.skip()

    assert Client.get_implementation("file://C:/bucket") == FileClient
    assert Client.get_implementation("file://C:\\bucket") == FileClient
    assert Client.get_implementation("file://\\bucket") == FileClient
    assert Client.get_implementation("file:///bucket") == FileClient
    assert Client.get_implementation("C://bucket") == FileClient
    assert Client.get_implementation("C:\\bucket") == FileClient
    assert Client.get_implementation("\bucket") == FileClient


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rel_path=non_null_text)
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
@given(rel_path=non_null_text)
def test_get_client(cloud_test_catalog, rel_path, cloud_type):
    catalog = cloud_test_catalog.catalog
    bucket_uri = cloud_test_catalog.src_uri
    url = f"{bucket_uri}/{rel_path}"
    client = Client.get_client(url, catalog.cache)
    assert client
    assert client.uri


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(rel_path=non_null_text)
def test_parse_url_uppercase_scheme(cloud_test_catalog, rel_path, cloud_type):
    if cloud_type == "file":
        assume(not rel_path.startswith("/"))
    bucket_uri = cloud_test_catalog.src_uri
    bucket_uri_upper = uppercase_scheme(bucket_uri)
    url = f"{bucket_uri_upper}/{rel_path}"
    uri, rel_part = Client.parse_url(url)
    if cloud_type == "file":
        url = f"{bucket_uri}/{rel_path}"
        assert uri == url.rsplit("/", 1)[0]
        assert rel_part == url.rsplit("/", 1)[1]
    else:
        assert uri == bucket_uri
        assert rel_part == rel_path


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


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_path_ends_with_slash(cloud_type):
    uri, rel_part = Client.parse_url("./animals/".replace("/", os.sep))
    assert uri == (Path().absolute() / Path("animals")).as_uri()
    assert rel_part == ""


@pytest.mark.parametrize("cloud_type", ["s3", "azure", "gs"], indirect=True)
def test_parse_cloud_path_ends_with_slash(cloud_test_catalog):
    uri = f"{cloud_test_catalog.src_uri}/animals/"
    uri, rel_part = Client.parse_url(uri)
    assert uri == cloud_test_catalog.src_uri
    assert rel_part == "animals/"
