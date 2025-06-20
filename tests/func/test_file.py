import pytest
import pytz

import datachain as dc
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.lib.file import File, FileError
from datachain.utils import TIME_ZERO


@pytest.mark.parametrize("cloud_type", ["s3"], indirect=True)
def test_get_path_cloud(cloud_test_catalog):
    file = File(path="dir/file", source="s3://")
    file._set_stream(catalog=cloud_test_catalog.catalog)
    assert file.get_fs_path().strip("/") == "s3:///dir/file"


@pytest.mark.parametrize("caching_enabled", [True, False])
def test_resolve_file(cloud_test_catalog, caching_enabled):
    ctc = cloud_test_catalog

    is_sqlite = isinstance(cloud_test_catalog.catalog.warehouse, SQLiteWarehouse)

    chain = dc.read_storage(ctc.src_uri, session=ctc.session)
    for orig_file in chain.collect("file"):
        file = File(
            source=orig_file.source,
            path=orig_file.path,
        )
        file._set_stream(catalog=ctc.catalog, caching_enabled=caching_enabled)
        resolved_file = file.resolve()
        if not is_sqlite:
            resolved_file.last_modified = resolved_file.last_modified.replace(
                microsecond=0, tzinfo=pytz.UTC
            )
        assert orig_file == resolved_file

        file.ensure_cached()


def test_resolve_file_no_exist(cloud_test_catalog):
    ctc = cloud_test_catalog

    non_existent_file = File(source=ctc.src_uri, path="non_existent_file.txt")
    non_existent_file._set_stream(catalog=ctc.catalog)
    resolved_non_existent = non_existent_file.resolve()
    assert resolved_non_existent.size == 0
    assert resolved_non_existent.etag == ""
    assert resolved_non_existent.last_modified == TIME_ZERO


@pytest.mark.parametrize("path", ["", ".", "..", "/", "dir/../../file.txt"])
def test_resolve_file_wrong_path(cloud_test_catalog, path):
    ctc = cloud_test_catalog

    wrong_file = File(source=ctc.src_uri, path=path)
    wrong_file._set_stream(catalog=ctc.catalog)
    resolved_wrong = wrong_file.resolve()
    assert resolved_wrong.size == 0
    assert resolved_wrong.etag == ""
    assert resolved_wrong.last_modified == TIME_ZERO


@pytest.mark.parametrize("caching_enabled", [True, False])
@pytest.mark.parametrize("path", ["", ".", "..", "/", "dir/../../file.txt"])
def test_cache_file_wrong_path(cloud_test_catalog, path, caching_enabled):
    ctc = cloud_test_catalog

    wrong_file = File(source=ctc.src_uri, path=path)
    wrong_file._set_stream(catalog=ctc.catalog, caching_enabled=caching_enabled)
    with pytest.raises(FileError):
        wrong_file.ensure_cached()


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
