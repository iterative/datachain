import json
from unittest.mock import Mock

import pytest
import pytz
from fsspec.implementations.local import LocalFileSystem
from PIL import Image

from datachain import DataChain
from datachain.catalog import Catalog
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.file import File, ImageFile, TextFile, resolve
from datachain.utils import TIME_ZERO


def create_file(source: str):
    return File(
        path="dir1/dir2/test.txt",
        source=source,
        etag="ed779276108738fdb2179ccabf9680d9",
    )


def test_file_stem():
    s = File(path=".file.jpg.txt")
    assert s.get_file_stem() == ".file.jpg"


def test_file_ext():
    s = File(path=".file.jpg.txt")
    assert s.get_file_ext() == "txt"


def test_file_suffix():
    s = File(path=".file.jpg.txt")
    assert s.get_file_suffix() == ".txt"


@pytest.mark.parametrize("name", [".file.jpg.txt", "dir1/dir2/name"])
def test_full_name(name):
    f = File(path=name)
    assert f.get_full_name() == name


def test_cache_get_path(catalog: Catalog):
    stream = File(path="test.txt1", source="s3://mybkt")
    stream._set_stream(catalog)

    data = b"some data is heRe"
    catalog.cache.store_data(stream, data)

    path = stream.get_local_path()
    assert path is not None

    with open(path, mode="rb") as f:
        assert f.read() == data


def test_get_destination_path_wrong_strategy():
    file = create_file("s3://mybkt")
    with pytest.raises(ValueError):
        file.get_destination_path("", "wrong")


def test_get_destination_path_filename_strategy():
    file = create_file("s3://mybkt")
    assert file.get_destination_path("output", "filename") == "output/test.txt"


def test_get_destination_path_empty_output():
    file = create_file("s3://mybkt")
    assert file.get_destination_path("", "filename") == "test.txt"


def test_get_destination_path_etag_strategy():
    file = create_file("s3://mybkt")
    assert (
        file.get_destination_path("output", "etag")
        == "output/ed779276108738fdb2179ccabf9680d9.txt"
    )


def test_get_destination_path_fullpath_strategy(catalog):
    file = create_file("s3://mybkt")
    file._set_stream(catalog, False)
    assert (
        file.get_destination_path("output", "fullpath")
        == "output/mybkt/dir1/dir2/test.txt"
    )


def test_get_destination_path_fullpath_strategy_file_source(catalog, tmp_path):
    file = create_file("file:///")
    file._set_stream(catalog, False)
    assert (
        file.get_destination_path("output", "fullpath") == "output/dir1/dir2/test.txt"
    )


def test_read_binary_data(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read() == data


def test_read_binary_data_as_text(tmp_path, catalog: Catalog):
    file_name = "myfile43.txt"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = TextFile(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    try:
        x = file.read()
    except UnicodeDecodeError:  # Unix
        pass
    else:  # Windows
        assert x != data


def test_read_text_data(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = TextFile(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    assert file.read() == data


def test_save_binary_data(tmp_path, catalog: Catalog):
    file1_name = "myfile1"
    file2_name = "myfile2"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    with open(tmp_path / file1_name, "wb") as fd:
        fd.write(data)

    file1 = File(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = File(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert file2.read() == data


def test_save_text_data(tmp_path, catalog: Catalog):
    file1_name = "myfile1.txt"
    file2_name = "myfile2.txt"
    data = "some text"

    with open(tmp_path / file1_name, "w") as fd:
        fd.write(data)

    file1 = TextFile(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = TextFile(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert file2.read() == data


def test_save_image_data(tmp_path, catalog: Catalog):
    from tests.utils import images_equal

    file1_name = "myfile1.jpg"
    file2_name = "myfile2.jpg"

    image = Image.new(mode="RGB", size=(64, 64))
    image.save(tmp_path / file1_name)

    file1 = ImageFile(path=file1_name, source=f"file://{tmp_path}")
    file1._set_stream(catalog, False)

    file1.save(tmp_path / file2_name)

    file2 = ImageFile(path=file2_name, source=f"file://{tmp_path}")
    file2._set_stream(catalog, False)
    assert images_equal(image, file2.read())


def test_cache_get_path_without_cache():
    stream = File(path="test.txt1", source="s3://mybkt")
    with pytest.raises(RuntimeError):
        stream.get_local_path()


def test_json_from_string():
    d = {"e": 12}

    file = File(path="something", location=d)
    assert file.location == d

    file = File(path="something", location=None)
    assert file.location is None

    file = File(path="something", location="")
    assert file.location is None

    file = File(path="something", location=json.dumps(d))
    assert file.location == d

    with pytest.raises(ValueError):
        File(path="something", location="{not a json}")


def test_file_info_jsons():
    file = File(path="something", location="")
    assert file.location is None

    d = {"e": 12}
    file = File(path="something", location=json.dumps(d))
    assert file.location == d


def test_get_path_local(catalog):
    file = File(path="dir/file", source="file:///")
    file._catalog = catalog
    assert file.get_path().replace("\\", "/").strip("/") == "dir/file"


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_get_path_cloud(cloud_test_catalog):
    file = File(path="dir/file", source="s3://")
    file._catalog = cloud_test_catalog.catalog
    assert file.get_path().strip("/") == "s3:///dir/file"


def test_get_fs(catalog):
    file = File(path="dir/file", source="file:///")
    file._catalog = catalog
    assert isinstance(file.get_fs(), LocalFileSystem)


def test_open_mode(tmp_path, catalog: Catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    with file.open(mode="r") as stream:
        assert stream.read() == data


def test_read_length(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read(length=4) == data[:4]


def test_read_bytes(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, False)
    assert file.read_bytes() == data


def test_read_text(tmp_path, catalog):
    file_name = "myfile"
    data = "this is a TexT data..."

    file_path = tmp_path / file_name
    with open(file_path, "w") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog, True)
    assert file.read_text() == data


def test_resolve_file(cloud_test_catalog):
    ctc = cloud_test_catalog

    is_sqlite = isinstance(cloud_test_catalog.catalog.warehouse, SQLiteWarehouse)

    dc = DataChain.from_storage(ctc.src_uri, session=ctc.session)
    for orig_file in dc.collect("file"):
        file = File(
            source=orig_file.source,
            path=orig_file.path,
        )
        file._catalog = ctc.catalog
        resolved_file = file.resolve()
        if not is_sqlite:
            resolved_file.last_modified = resolved_file.last_modified.replace(
                microsecond=0, tzinfo=pytz.UTC
            )
        assert orig_file == resolved_file


def test_resolve_file_no_exist(cloud_test_catalog):
    ctc = cloud_test_catalog

    non_existent_file = File(source=ctc.src_uri, path="non_existent_file.txt")
    non_existent_file._catalog = ctc.catalog
    resolved_non_existent = non_existent_file.resolve()
    assert resolved_non_existent.size == 0
    assert resolved_non_existent.etag == ""
    assert resolved_non_existent.last_modified == TIME_ZERO


def test_resolve_unsupported_protocol():
    mock_catalog = Mock()
    mock_catalog.get_client.side_effect = NotImplementedError("Unsupported protocol")

    file = File(source="unsupported://example.com", path="test.txt")
    file._catalog = mock_catalog

    with pytest.raises(RuntimeError) as exc_info:
        file.resolve()

    assert (
        str(exc_info.value)
        == "Unsupported protocol for file source: unsupported://example.com"
    )


def test_file_resolve_no_catalog():
    file = File(path="test.txt", source="s3://mybucket")
    with pytest.raises(RuntimeError, match="Cannot resolve file: catalog is not set"):
        file.resolve()


def test_resolve_function():
    mock_file = Mock(spec=File)
    mock_file.resolve.return_value = "resolved_file"

    result = resolve(mock_file)

    assert result == "resolved_file"
    mock_file.resolve.assert_called_once()


def test_get_local_path(tmp_path, catalog):
    file_name = "myfile"
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"

    file_path = tmp_path / file_name
    with open(file_path, "wb") as fd:
        fd.write(data)

    file = File(path=file_name, source=f"file://{tmp_path}")
    file._set_stream(catalog)

    assert file.get_local_path() is None
    file.ensure_cached()
    assert file.get_local_path() is not None
