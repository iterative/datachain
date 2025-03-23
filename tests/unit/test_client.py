import os
import sys
from pathlib import Path

import pytest

from datachain.client import Client
from datachain.client.local import FileClient


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


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_parse_file_path_ends_with_slash(cloud_type):
    uri, rel_part = Client.parse_url("./animals/".replace("/", os.sep))
    assert uri == (Path().absolute() / Path("animals")).as_uri()
    assert rel_part == ""
