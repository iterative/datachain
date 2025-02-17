import pytest
import pytz

from datachain import DataChain
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.file import File
from datachain.utils import TIME_ZERO


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_get_path_cloud(cloud_test_catalog):
    file = File(path="dir/file", source="s3://")
    file._catalog = cloud_test_catalog.catalog
    assert file.get_path().strip("/") == "s3:///dir/file"


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


def test_upload(cloud_test_catalog):
    ctc = cloud_test_catalog

    src_uri = ctc.src_uri
    filename = "image_1.jpg"
    dest = f"{src_uri}/upload-test-images"
    catalog = ctc.catalog

    img_bytes = b"bytes"

    f = File.upload(img_bytes, f"{dest}/{filename}", catalog)

    client = catalog.get_client(src_uri)
    source, rel_path = client.split_url(f"{dest}/{filename}")

    assert f.path == rel_path
    assert f.source == client.get_uri(source)
    assert f.read() == img_bytes

    client.fs.rm(dest, recursive=True)
