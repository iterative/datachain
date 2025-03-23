from tarfile import TarInfo

import pytest

from datachain.lib.file import File
from datachain.lib.webdataset import (
    CoreFileDuplicationError,
    CoreFileNotFoundError,
    UnknownFileExtensionError,
    get_tar_groups,
)
from datachain.lib.webdataset_laion import WDSLaion


class MockTarInfo(TarInfo):
    def __init__(self, name, content=b"", size=0, offset=0):
        super().__init__(name)
        self._content = content.encode() if isinstance(content, str) else content
        self.size = size
        self.offset = offset

    def isfile(self):
        return True


class MockTar:
    def __init__(self, members: list[MockTarInfo]):
        self._members = members

    def getmembers(self):
        return self._members

    def extractfile(self, tar_info: MockTarInfo):
        class TmpReader:
            def read(self):
                return tar_info._content

        return TmpReader()


def test_webdataset_basic():
    tar_file = File(path="nnn.tar")
    tar = MockTar(
        [
            MockTarInfo("01.jpg"),
            MockTarInfo("01.json", b'{"uid": "5678"}'),
            MockTarInfo("64.jpg"),
            MockTarInfo("64.json", b"{}"),
        ]
    )

    groups = list(get_tar_groups(tar_file, tar, ["jpg"], WDSLaion))

    assert len(groups) == 2

    laion01, laion64 = groups

    assert laion01.file.name == "01.jpg"
    assert laion01.file.parent == tar_file.name
    assert laion01.file.location is not None
    assert isinstance(laion01.file.location, list)
    assert len(laion01.file.location) > 0

    parent = laion01.file.location[0].get("parent", None)
    assert parent is not None
    parent_file = File(**parent)
    assert parent_file == tar_file

    assert laion64.file.name == "64.jpg"
    assert laion64.file.parent == tar_file.name

    assert laion01.txt is None
    assert laion01.json.uid == "5678"

    assert laion64.json.uid == ""


def test_webdataset_empty():
    stream = File(path="nnn.tar")
    tar = MockTar([])

    groups = list(get_tar_groups(stream, tar, ["jpg"], WDSLaion))

    assert len(groups) == 0


def test_webdataset_missing_core_files():
    stream = File(path="nnn.tar")
    tar = MockTar(
        [
            MockTarInfo("01.txt"),
            MockTarInfo("01.json", b'{"uid": "5678"}'),
            MockTarInfo("64.txt"),
            MockTarInfo("64.json", b"{}"),
        ]
    )

    with pytest.raises(CoreFileNotFoundError):
        list(get_tar_groups(stream, tar, ["NONSENSE"], WDSLaion))


def test_webdataset_single_file_per_group():
    stream = File(path="nnn.tar")
    tar = MockTar(
        [
            MockTarInfo("01.jpg"),
            MockTarInfo("64.jpg"),
            MockTarInfo("64.json", b"{}"),
            MockTarInfo("03.jpg"),
        ]
    )

    groups = list(get_tar_groups(stream, tar, ["jpg"], WDSLaion))
    assert len(groups) == 3


def test_webdataset_multiple_core_extensions():
    stream = File(path="nnn.tar")
    tar = MockTar(
        [
            MockTarInfo("01.png"),
            MockTarInfo("64.png"),
            MockTarInfo("64.json", b"{}"),
            MockTarInfo("03.jpg"),
        ]
    )

    groups = list(get_tar_groups(stream, tar, ["jpg", "png"], WDSLaion))
    assert len(groups) == 3


def test_webdataset_core_file_duplication():
    stream = File(path="nnn.tar")
    tar = MockTar(
        [
            MockTarInfo("01.png"),
            MockTarInfo("01.jpg"),
            MockTarInfo("64.json", b"{}"),
            MockTarInfo("03.jpg"),
        ]
    )

    with pytest.raises(CoreFileDuplicationError):
        list(get_tar_groups(stream, tar, ["jpg", "png"], WDSLaion))


def test_webdataset_unknown_file_type():
    stream = File(path="nnn.tar")
    tar = MockTar([MockTarInfo("01.QQQQ")])

    with pytest.raises(UnknownFileExtensionError):
        list(get_tar_groups(stream, tar, ["jpg", "png"], WDSLaion))
