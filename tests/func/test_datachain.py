import math
import os
import pickle
import posixpath
import re
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz
from PIL import Image
from sqlalchemy import Column

from datachain.catalog.catalog import QUERY_SCRIPT_CANCELED_EXIT_CODE
from datachain.client.local import FileClient
from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.dataset import DatasetDependencyType, DatasetStats
from datachain.lib.dc import C, DataChain, DataChainColumnError
from datachain.lib.file import File, ImageFile
from datachain.lib.listing import (
    LISTING_TTL,
    is_listing_dataset,
    parse_listing_uri,
)
from datachain.lib.tar import process_tar
from datachain.lib.udf import Mapper
from datachain.lib.utils import DataChainError
from datachain.query.dataset import QueryStep
from datachain.sql.functions import path as pathfunc
from datachain.sql.functions.array import cosine_distance, euclidean_distance
from tests.utils import NUM_TREE, TARRED_TREE, images_equal, text_embedding


def _get_listing_datasets(session):
    return sorted(
        [
            f"{ds.name}@v{ds.version}"
            for ds in DataChain.datasets(session=session, include_listing=True).collect(
                "dataset"
            )
            if is_listing_dataset(ds.name)
        ]
    )


@pytest.mark.parametrize("anon", [True, False])
def test_catalog_anon(tmp_dir, catalog, anon):
    chain = DataChain.from_storage(tmp_dir.as_uri(), anon=anon)
    assert chain.session.catalog.client_config.get("anon", False) is anon


def test_from_storage(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(ctc.src_uri, session=ctc.session)
    assert dc.count() == 7


def test_from_storage_non_recursive(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(
        f"{ctc.src_uri}/dogs", session=ctc.session, recursive=False
    )
    assert dc.count() == 3


def test_from_storage_glob(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(f"{ctc.src_uri}/dogs*", session=ctc.session)
    assert dc.count() == 4


def test_from_storage_as_image(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(ctc.src_uri, session=ctc.session, type="image")
    for im in dc.collect("file"):
        assert isinstance(im, ImageFile)


def test_from_storage_reindex(tmp_dir, test_session):
    tmp_dir = tmp_dir / "parquets"
    path = tmp_dir.as_uri()
    os.mkdir(tmp_dir)

    pd.DataFrame({"name": ["Alice", "Bob"]}).to_parquet(tmp_dir / "test1.parquet")
    assert DataChain.from_storage(path, session=test_session).count() == 1

    pd.DataFrame({"name": ["Charlie", "David"]}).to_parquet(tmp_dir / "test2.parquet")
    assert DataChain.from_storage(path, session=test_session).count() == 1
    assert DataChain.from_storage(path, session=test_session, update=True).count() == 2


def test_from_storage_reindex_expired(tmp_dir, test_session):
    catalog = test_session.catalog
    tmp_dir = tmp_dir / "parquets"
    os.mkdir(tmp_dir)
    uri = tmp_dir.as_uri()

    lst_ds_name = parse_listing_uri(uri, catalog.cache, catalog.client_config)[0]

    pd.DataFrame({"name": ["Alice", "Bob"]}).to_parquet(tmp_dir / "test1.parquet")
    assert DataChain.from_storage(uri, session=test_session).count() == 1
    pd.DataFrame({"name": ["Charlie", "David"]}).to_parquet(tmp_dir / "test2.parquet")
    # mark dataset as expired
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset(lst_ds_name),
        1,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=LISTING_TTL + 20),
    )

    # listing was updated because listing dataset was expired
    assert DataChain.from_storage(uri, session=test_session).count() == 2


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_from_storage_partials(cloud_test_catalog):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    session = ctc.session
    catalog = session.catalog

    def _list_dataset_name(uri: str) -> str:
        return parse_listing_uri(uri, catalog.cache, catalog.client_config)[0]

    dogs_uri = f"{src_uri}/dogs"
    DataChain.from_storage(dogs_uri, session=session)
    assert _get_listing_datasets(session) == [
        f"{_list_dataset_name(dogs_uri)}@v1",
    ]

    DataChain.from_storage(f"{src_uri}/dogs/others", session=session)
    assert _get_listing_datasets(session) == [
        f"{_list_dataset_name(dogs_uri)}@v1",
    ]

    DataChain.from_storage(src_uri, session=session)
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(dogs_uri)}@v1",
            f"{_list_dataset_name(src_uri)}@v1",
        ]
    )

    DataChain.from_storage(f"{src_uri}/cats", session=session)
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(dogs_uri)}@v1",
            f"{_list_dataset_name(src_uri)}@v1",
        ]
    )


@pytest.mark.parametrize(
    "cloud_type",
    ["s3", "azure", "gs"],
    indirect=True,
)
def test_from_storage_partials_with_update(cloud_test_catalog):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    session = ctc.session
    catalog = session.catalog

    def _list_dataset_name(uri: str) -> str:
        return parse_listing_uri(uri, catalog.cache, catalog.client_config)[0]

    uri = f"{src_uri}/cats"
    DataChain.from_storage(uri, session=session)
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(uri)}@v1",
        ]
    )

    DataChain.from_storage(uri, session=session, update=True)
    assert _get_listing_datasets(session) == sorted(
        [
            f"{_list_dataset_name(uri)}@v1",
            f"{_list_dataset_name(uri)}@v2",
        ]
    )


def test_from_storage_dependencies(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    src_uri = ctc.src_uri
    uri = f"{src_uri}/cats"
    ds_name = "dep"
    DataChain.from_storage(uri, session=ctc.session).save(ds_name)
    dependencies = ctc.session.catalog.get_dataset_dependencies(ds_name, 1)
    assert len(dependencies) == 1
    assert dependencies[0].type == DatasetDependencyType.STORAGE
    if cloud_type == "file":
        assert dependencies[0].name == FileClient.root_path().as_uri()
    else:
        assert dependencies[0].name == src_uri


@pytest.mark.parametrize("use_cache", [True, False])
def test_map_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    dc = (
        DataChain.from_storage(ctc.src_uri, session=ctc.session)
        .settings(cache=use_cache)
        .map(signal=new_signal)
    )

    expected = {
        "description -> Cats and Dogs",
        "cat1 -> meow",
        "cat2 -> mrow",
        "dog1 -> woof",
        "dog2 -> arf",
        "dog3 -> bark",
        "dog4 -> ruff",
    }
    assert set(dc.collect("signal")) == expected
    for file in dc.collect("file"):
        assert bool(file.get_local_path()) is use_cache


@pytest.mark.parametrize("use_cache", [False, True])
def test_read_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    dc = DataChain.from_storage(ctc.src_uri, session=ctc.session)
    for file in dc.settings(cache=use_cache).collect("file"):
        assert file.get_local_path() is None
        file.read()
        assert bool(file.get_local_path()) is use_cache


@pytest.mark.parametrize("placement", ["fullpath", "filename"])
@pytest.mark.parametrize("use_map", [True, False])
@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("file_type", ["", "binary", "text"])
@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_export_files(
    tmp_dir, cloud_test_catalog, test_session, placement, use_map, use_cache, file_type
):
    ctc = cloud_test_catalog
    df = DataChain.from_storage(ctc.src_uri, type=file_type, session=test_session)
    if use_map:
        df.export_files(tmp_dir / "output", placement=placement, use_cache=use_cache)
        df.map(
            res=lambda file: file.export(
                tmp_dir / "output", placement=placement, use_cache=use_cache
            )
        ).exec()
    else:
        df.export_files(tmp_dir / "output", placement=placement)

    expected = {
        "description": "Cats and Dogs",
        "cat1": "meow",
        "cat2": "mrow",
        "dog1": "woof",
        "dog2": "arf",
        "dog3": "bark",
        "dog4": "ruff",
    }

    for file in df.collect("file"):
        if placement == "filename":
            file_path = file.name
        else:
            file_path = file.get_full_name()
        with open(tmp_dir / "output" / file_path) as f:
            assert f.read() == expected[file.name]


@pytest.mark.parametrize("use_cache", [True, False])
def test_export_images_files(test_session, tmp_dir, tmp_path, use_cache):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    DataChain.from_values(
        file=[
            ImageFile(path=img["name"], source=f"file://{tmp_path}") for img in images
        ],
        session=test_session,
    ).export_files(tmp_dir / "output", placement="filename", use_cache=use_cache)

    for img in images:
        exported_img = Image.open(tmp_dir / "output" / img["name"])
        assert images_equal(img["data"], exported_img)


def test_export_files_filename_placement_not_unique_files(tmp_dir, test_session):
    data = b"some\x00data\x00is\x48\x65\x6c\x57\x6f\x72\x6c\x64\xff\xffheRe"
    bucket_name = "mybucket"
    files = ["dir1/a.json", "dir1/dir2/a.json"]

    # create bucket dir with duplicate file names
    bucket_dir = tmp_dir / bucket_name
    bucket_dir.mkdir(parents=True)
    for file_path in files:
        file_path = bucket_dir / file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as fd:
            fd.write(data)

    df = DataChain.from_storage((tmp_dir / bucket_name).as_uri(), session=test_session)
    with pytest.raises(ValueError):
        df.export_files(tmp_dir / "output", placement="filename")


def test_show(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    DataChain.from_values(
        first_name=first_name,
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "first_name age city" in normalized_output
    for i in range(3):
        assert f"{i} {first_name[i]}" in normalized_output


def test_show_nested_empty(capsys, test_session):
    files = [File(size=s, path=p) for p, s in zip(list("abcde"), range(5))]
    DataChain.from_values(file=files, session=test_session).limit(0).show()

    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "Empty result" in normalized_output
    assert "('file', 'path')" in normalized_output


def test_show_empty(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    DataChain.from_values(first_name=first_name, session=test_session).limit(0).show()

    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "Empty result" in normalized_output
    assert "Columns: ['first_name']" in normalized_output


def test_show_limit(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    DataChain.from_values(
        first_name=first_name,
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
        session=test_session,
    ).limit(1).show()
    captured = capsys.readouterr()
    new_line_count = captured.out.count("\n")
    assert new_line_count == 2


def test_show_transpose(capsys, test_session):
    first_name = ["Alice", "Bob", "Charlie"]
    last_name = ["A", "B", "C"]
    DataChain.from_values(
        first_name=first_name,
        last_name=last_name,
        session=test_session,
    ).show(transpose=True)
    captured = capsys.readouterr()
    stripped_output = re.sub(r"\s+", " ", captured.out)
    assert " ".join(first_name) in stripped_output
    assert " ".join(last_name) in stripped_output


def test_show_truncate(capsys, test_session):
    client = ["Alice A", "Bob B", "Charles C"]
    details = [
        "This is a very long piece of text that would not fit in the default output "
        "because pandas will truncate the column",
        "Gives good tips",
        "Not very nice",
    ]

    dc = DataChain.from_values(
        client=client,
        details=details,
        session=test_session,
    )

    dc.show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert f"{client[0]} {details[0][:10]}" in normalized_output
    assert details[0] not in normalized_output
    for i in [1, 2]:
        assert f"{client[i]} {details[i]}" in normalized_output


def test_show_no_truncate(capsys, test_session):
    client = ["Alice A", "Bob B", "Charles C"]
    details = [
        "This is a very long piece of text that would not fit in the default output "
        "because pandas will truncate the column",
        "Gives good tips",
        "Not very nice",
    ]

    dc = DataChain.from_values(
        client=client,
        details=details,
        session=test_session,
    )

    dc.show(truncate=False)
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    for i in range(3):
        assert client[i] in normalized_output
        assert details[i] in normalized_output


def test_from_storage_dataset_stats(tmp_dir, test_session):
    for i in range(4):
        (tmp_dir / f"file{i}.txt").write_text(f"file{i}")

    dc = DataChain.from_storage(tmp_dir.as_uri(), session=test_session).save(
        "test-data"
    )
    stats = test_session.catalog.dataset_stats(dc.name, dc.version)
    assert stats == DatasetStats(num_objects=4, size=20)


def test_from_storage_check_rows(tmp_dir, test_session):
    stats = {}
    for i in range(4):
        file = tmp_dir / f"{i}.txt"
        file.write_text(f"file{i}")
        stats[file.name] = file.stat()

    dc = DataChain.from_storage(tmp_dir.as_uri(), session=test_session).save(
        "test-data"
    )

    is_sqlite = isinstance(test_session.catalog.warehouse, SQLiteWarehouse)
    tz = timezone.utc if is_sqlite else pytz.UTC

    for (file,) in dc.collect():
        assert isinstance(file, File)
        stat = stats[file.name]
        mtime = stat.st_mtime if is_sqlite else float(math.floor(stat.st_mtime))
        assert file == File(
            source=Path(tmp_dir.anchor).as_uri(),
            path=file.path,
            size=stat.st_size,
            version="",
            etag=stat.st_mtime.hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(mtime, tz=tz),
            location=None,
        )


def test_mutate_existing_column(test_session):
    ds = DataChain.from_values(ids=[1, 2, 3], session=test_session)

    with pytest.raises(DataChainColumnError) as excinfo:
        ds.mutate(ids=Column("ids") + 1)

    assert (
        str(excinfo.value)
        == "Error for column ids: Cannot modify existing column with mutate()."
        " Use a different name for the new column."
    )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    def name_len(path):
        return (len(posixpath.basename(path)),)

    dc = (
        DataChain.from_storage(cloud_test_catalog.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .map(name_len, params=["file.path"], output={"name_len": int})
    )
    result1 = list(dc.select("file.path", "name_len").collect())
    # ensure that we're able to run with same query multiple times
    result2 = list(dc.select("file.path", "name_len").collect())
    count = dc.count()
    assert len(result1) == 3
    assert len(result2) == 3
    assert count == 3

    for r1, r2 in zip(result1, result2):
        # Check that the UDF ran successfully
        assert len(posixpath.basename(r1[0])) == r1[1]
        assert len(posixpath.basename(r2[0])) == r2[1]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name):
        return (len(name),)

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=-1)
        .map(name_len, params=["file.path"], output={"name_len": int})
        .select("file.path", "name_len")
    )

    # Check that the UDF ran successfully
    count = 0
    for r in dc.collect():
        count += 1
        assert len(r[0]) == r[1]
    assert count == 7


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed(cloud_test_catalog_tmpfile, workers, datachain_job_id):
    session = cloud_test_catalog_tmpfile.session

    def name_len(name):
        return (len(name),)

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=2, workers=workers)
        .map(name_len, params=["file.path"], output={"name_len": int})
        .select("file.path", "name_len")
    )

    # Check that the UDF ran successfully
    count = 0
    for r in dc.collect():
        count += 1
        assert len(r[0]) == r[1]
    assert count == 7


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_class_udf(cloud_test_catalog):
    session = cloud_test_catalog.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    dc = (
        DataChain.from_storage(cloud_test_catalog.src_uri, session=session)
        .filter(C("file.size") < 13)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert list(dc.collect()) == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_class_udf_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    class MyUDF(Mapper):
        def __init__(self, constant, multiplier=1):
            self.constant = constant
            self.multiplier = multiplier

        def process(self, size):
            return (self.constant + size * self.multiplier,)

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .settings(parallel=2)
        .map(
            MyUDF(5, multiplier=2),
            output={"total": int},
            params=["file.size"],
        )
        .select("file.size", "total")
        .order_by("file.size")
    )

    assert list(dc.collect()) == [
        (3, 11),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
        (4, 13),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_exec_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .settings(parallel=-1)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        dc.show()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.parametrize("workers", (1, 2))
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_exec_error(
    cloud_test_catalog_tmpfile, workers, datachain_job_id
):
    session = cloud_test_catalog_tmpfile.session

    def name_len_error(_name):
        # A udf that raises an exception
        raise RuntimeError("Test Error!")

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .settings(parallel=2, workers=workers)
        .map(name_len_error, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(DataChainError, match="Test Error!"):
        dc.show()


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_reuse_on_error(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    error_state = {"error": True}

    def name_len_maybe_error(path):
        if error_state["error"]:
            # A udf that raises an exception
            raise RuntimeError("Test Error!")
        return (len(path),)

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .map(name_len_maybe_error, params=["file.path"], output={"path_len": int})
        .select("file.path", "path_len")
    )

    with pytest.raises(DataChainError, match="Test Error!"):
        dc.show()

    # Simulate fixing the error
    error_state["error"] = False

    # Retry Query
    count = 0
    for r in dc.collect():
        # Check that the UDF ran successfully
        count += 1
        assert len(r[0]) == r[1]
    assert count == 3


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_parallel_interrupt(cloud_test_catalog_tmpfile, capfd):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .settings(parallel=-1)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(RuntimeError, match="UDF Execution Failed!"):
        dc.show()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_interrupt(cloud_test_catalog_tmpfile, capfd, datachain_job_id):
    session = cloud_test_catalog_tmpfile.session

    def name_len_interrupt(_name):
        # A UDF that emulates cancellation due to a KeyboardInterrupt.
        raise KeyboardInterrupt

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .settings(parallel=2, workers=2)
        .map(name_len_interrupt, params=["file.path"], output={"name_len": int})
    )
    with pytest.raises(RuntimeError, match=r"Worker Killed \(KeyboardInterrupt\)"):
        dc.show()
    captured = capfd.readouterr()
    assert "KeyboardInterrupt" in captured.err
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.skipif(
    "not os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Set the DATACHAIN_DISTRIBUTED environment variable "
    "to test distributed UDFs",
)
@pytest.mark.xdist_group(name="tmpfile")
def test_udf_distributed_cancel(cloud_test_catalog_tmpfile, capfd, datachain_job_id):
    catalog = cloud_test_catalog_tmpfile.catalog
    session = cloud_test_catalog_tmpfile.session
    metastore = catalog.metastore

    job_id = os.environ.get("DATACHAIN_JOB_ID")

    # A job is required for query script cancellation (not using a KeyboardInterrupt)
    metastore.db.execute(
        metastore._jobs_insert().values(
            id=job_id,
            status=7,  # CANCELING
            celery_task_id="",
            name="Test Cancel Job",
            workers=2,
            team_id=metastore.team_id,
            created_at=datetime.now(timezone.utc),
            params="{}",
            metrics="{}",
        ),
    )

    def name_len_slow(name):
        # A very simple udf, that processes slowly to emulate being stuck.
        from time import sleep

        sleep(10)
        return len(name), None

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .filter(C("file.size") < 13)
        .filter(C("file.path").glob("cats*") | (C("file.size") < 4))
        .settings(parallel=2, workers=2)
        .map(name_len_slow, params=["file.path"], output={"name_len": int})
    )

    with pytest.raises(SystemExit) as excinfo:
        dc.show()

    assert excinfo.value.code == QUERY_SCRIPT_CANCELED_EXIT_CODE
    captured = capfd.readouterr()
    assert "canceled" in captured.out
    assert "semaphore" not in captured.err


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_avoid_recalculation_after_save(cloud_test_catalog):
    calls = 0

    def name_len(path):
        nonlocal calls
        calls += 1
        return (len(path),)

    uri = cloud_test_catalog.src_uri
    session = cloud_test_catalog.session
    ds = (
        DataChain.from_storage(uri, session=session)
        .filter(C("file.path").glob("*/dog1"))
        .map(name_len, params=["file.path"], output={"name_len": int})
    )
    ds2 = ds.save("ds1")

    assert ds2._query.steps == []
    assert ds2._query.dependencies == set()
    assert isinstance(ds2._query.starting_step, QueryStep)
    ds2.save("ds2")
    assert calls == 1  # UDF should be called only once


@pytest.mark.parametrize(
    "cloud_type,version_aware,tree",
    [("s3", True, NUM_TREE), ("file", False, NUM_TREE)],
    indirect=True,
)
def test_udf_after_limit(cloud_test_catalog):
    ctc = cloud_test_catalog

    def name_int(name: str) -> int:
        try:
            return int(name)
        except ValueError:
            return 0

    def get_result(chain):
        res = chain.limit(100).map(name_int=name_int).order_by("name")
        return list(res.collect("name", "name_int"))

    expected = [(f"{i:06d}", i) for i in range(100)]
    dc = (
        DataChain.from_storage(ctc.src_uri, session=ctc.session)
        .mutate(name=pathfunc.name(C("file.path")))
        .save()
    )
    # We test a few different orderings here, because we've had strange
    # bugs in the past where calling add_signals() after limit() gave us
    # incorrect results on clickhouse cloud.
    # See https://github.com/iterative/dvcx/issues/940
    assert get_result(dc.order_by("name")) == expected
    assert len(get_result(dc.order_by("sys.rand"))) == 100
    assert len(get_result(dc)) == 100


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_name_len_desc_and_name_asc(cloud_test_catalog):
    session = cloud_test_catalog.session

    path = cloud_test_catalog.src_uri
    ds_name = uuid.uuid4().hex

    def name_len(path):
        return (len(posixpath.basename(path)),)

    DataChain.from_storage(path, session=session).map(
        name_len, params=["file.path"], output={"name_len": int}
    ).order_by("name_len", descending=True).order_by("file.path").save(ds_name)

    assert list(
        DataChain.from_dataset(name=ds_name, session=session).collect(
            "sys.id", "file.path"
        )
    ) == [
        (1, "description"),
        (2, "cats/cat1"),
        (3, "cats/cat2"),
        (4, "dogs/dog1"),
        (5, "dogs/dog2"),
        (6, "dogs/dog3"),
        (7, "dogs/others/dog4"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_row_number_with_order_by_before_map(cloud_test_catalog):
    session = cloud_test_catalog.session

    path = cloud_test_catalog.src_uri
    ds_name = uuid.uuid4().hex

    def name_len(path):
        return (len(posixpath.basename(path)),)

    DataChain.from_storage(path, session=session).order_by("file.path").map(
        name_len, params=["file.path"], output={"name_len": int}
    ).save(ds_name)

    # we should preserve order in final result based on order by which was added
    # before add_signals
    assert list(
        DataChain.from_dataset(name=ds_name, session=session).collect(
            "sys.id", "file.path"
        )
    ) == [
        (1, "cats/cat1"),
        (2, "cats/cat2"),
        (3, "description"),
        (4, "dogs/dog1"),
        (5, "dogs/dog2"),
        (6, "dogs/dog3"),
        (7, "dogs/others/dog4"),
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_udf_different_types(cloud_test_catalog):
    obj = {"name": "John", "age": 30}

    def test_types():
        return (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            pickle.dumps(obj),
        )

    dc = (
        DataChain.from_storage(
            cloud_test_catalog.src_uri, session=cloud_test_catalog.session
        )
        .filter(C("file.path").glob("*cat1"))
        .map(
            test_types,
            params=[],
            output={
                "int_col": int,
                "int_col_32": int,
                "int_col_64": int,
                "float_col": float,
                "float_col_32": float,
                "float_col_64": float,
                "array_col": list[float],
                "array_col_nested": list[list[float]],
                "array_col_32": list[float],
                "array_col_64": list[float],
                "string_col": str,
                "bool_col": bool,
                "json_col": dict,
                "binary_col": bytes,
            },
        )
    )

    results = dc.to_records()
    col_values = [
        (
            r["int_col"],
            r["int_col_32"],
            r["int_col_64"],
            r["float_col"],
            r["float_col_32"],
            r["float_col_64"],
            r["array_col"],
            r["array_col_nested"],
            r["array_col_32"],
            r["array_col_64"],
            r["string_col"],
            r["bool_col"],
            r["json_col"],
            pickle.loads(r["binary_col"]),  # noqa: S301
        )
        for r in results
    ]

    assert col_values == [
        (
            5,
            5,
            5,
            0.5,
            0.5,
            0.5,
            [0.5],
            [[0.5], [0.5]],
            [0.5],
            [0.5],
            "s",
            True,
            {"a": 1},
            obj,
        )
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
@pytest.mark.xdist_group(name="tmpfile")
def test_gen_parallel(cloud_test_catalog_tmpfile):
    session = cloud_test_catalog_tmpfile.session

    def func(file) -> Iterator[tuple[str]]:
        for i in range(5):
            yield (f"{file.path}_{i}",)

    dc = (
        DataChain.from_storage(cloud_test_catalog_tmpfile.src_uri, session=session)
        .settings(parallel=-1)
        .gen(gen=func, params=["file"], output={"val": str})
        .order_by("val")
    )
    assert list(dc.collect("val")) == [
        "cats/cat1_0",
        "cats/cat1_1",
        "cats/cat1_2",
        "cats/cat1_3",
        "cats/cat1_4",
        "cats/cat2_0",
        "cats/cat2_1",
        "cats/cat2_2",
        "cats/cat2_3",
        "cats/cat2_4",
        "description_0",
        "description_1",
        "description_2",
        "description_3",
        "description_4",
        "dogs/dog1_0",
        "dogs/dog1_1",
        "dogs/dog1_2",
        "dogs/dog1_3",
        "dogs/dog1_4",
        "dogs/dog2_0",
        "dogs/dog2_1",
        "dogs/dog2_2",
        "dogs/dog2_3",
        "dogs/dog2_4",
        "dogs/dog3_0",
        "dogs/dog3_1",
        "dogs/dog3_2",
        "dogs/dog3_3",
        "dogs/dog3_4",
        "dogs/others/dog4_0",
        "dogs/others/dog4_1",
        "dogs/others/dog4_2",
        "dogs/others/dog4_3",
        "dogs/others/dog4_4",
    ]


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_gen_with_new_columns_numpy(cloud_test_catalog, dogs_dataset):
    session = cloud_test_catalog.session

    def gen_numpy():
        for _ in range(10):
            yield (
                np.int32(11),
                np.int64(12),
                np.float32(0.5),
                np.float64(0.5),
                np.int32(13),
                np.array([[0.5], [0.5]], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float32),
                np.array([0.5, 0.5], dtype=np.float64),
                np.array([14, 15], dtype=np.int32),
                np.array([], dtype=np.float32),
            )

    DataChain.from_storage(cloud_test_catalog.src_uri, session=session).gen(
        subobject=gen_numpy,
        output={
            "int_col_32": int,
            "int_col_64": int,
            "float_col_32": float,
            "float_col_64": float,
            "int_float_col_32": float,
            "array_col_nested": list[list[float]],
            "array_col_32": list[float],
            "array_col_64": list[float],
            "array_int_float_col_32": list[float],
            "array_empty_col_32": list[float],
        },
    ).save("dogs_with_rows_and_signals")

    dc = DataChain.from_dataset(name="dogs_with_rows_and_signals", session=session)
    for r in dc.collect(
        "int_col_32",
        "int_col_64",
        "float_col_32",
        "float_col_64",
        "int_float_col_32",
        "array_col_nested",
        "array_col_32",
        "array_col_64",
        "array_int_float_col_32",
        "array_empty_col_32",
    ):
        assert r == (
            11,
            12,
            0.5,
            0.5,
            13.0,
            [[0.5], [0.5]],
            [0.5, 0.5],
            [0.5, 0.5],
            [14.0, 15.0],
            [],
        )


@pytest.mark.parametrize(
    "cloud_type,version_aware",
    [("s3", True)],
    indirect=True,
)
def test_gen_with_new_columns_wrong_type(cloud_test_catalog, dogs_dataset):
    session = cloud_test_catalog.session

    def gen_func():
        yield (0.5)

    with pytest.raises(ValueError):
        DataChain.from_storage(cloud_test_catalog.src_uri, session=session).gen(
            new_val=gen_func, output={"new_val": int}
        ).show()


def test_similarity_search(cloud_test_catalog):
    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    def calc_emb(file):
        text = file.read().decode("utf-8")
        return text_embedding(text)

    target_embedding = next(
        DataChain.from_storage(src_uri, session=session)
        .filter(C("file.path").glob("*description"))
        .order_by("file.path")
        .limit(1)
        .map(embedding=calc_emb, output={"embedding": list[float]})
        .collect("embedding")
    )
    dc = (
        DataChain.from_storage(src_uri, session=session)
        .map(embedding=calc_emb, output={"embedding": list[float]})
        .mutate(
            cos_dist=cosine_distance(C("embedding"), target_embedding),
            eucl_dist=euclidean_distance(C("embedding"), target_embedding),
        )
        .order_by("file.path")
    )
    count = dc.count()
    assert count == 7

    expected = [
        ("cats/cat1", 0.8508677010357059, 1.9078358385397216),
        ("cats/cat2", 0.8508677010357059, 1.9078358385397216),
        ("description", 0.0, 0.0),
        ("dogs/dog1", 0.7875133863812602, 1.8750659656122843),
        ("dogs/dog2", 0.7356502722055684, 1.775619888314893),
        ("dogs/dog3", 0.7695916496857775, 1.8344983482620636),
        ("dogs/others/dog4", 0.9789704524691446, 2.0531542018152322),
    ]

    for (p1, c1, e1), (p2, c2, e2) in zip(
        dc.collect("file.path", "cos_dist", "eucl_dist"), expected
    ):
        assert p1.endswith(p2)
        assert math.isclose(c1, c2, abs_tol=1e-5)
        assert math.isclose(e1, e2, abs_tol=1e-5)


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
def test_process_and_open_tar(cloud_test_catalog, cloud_type):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(ctc.src_uri, session=ctc.session).gen(file=process_tar)
    assert dc.count() == 7

    if cloud_type == "file":
        prefix = cloud_test_catalog.partial_path + "/"
    else:
        prefix = ""

    assert {(file.read(), file.path) for file in dc.collect("file")} == {
        (b"meow", f"{prefix}animals.tar/cats/cat1"),
        (b"mrow", f"{prefix}animals.tar/cats/cat2"),
        (b"Cats and Dogs", f"{prefix}animals.tar/description"),
        (b"woof", f"{prefix}animals.tar/dogs/dog1"),
        (b"arf", f"{prefix}animals.tar/dogs/dog2"),
        (b"bark", f"{prefix}animals.tar/dogs/dog3"),
        (b"ruff", f"{prefix}animals.tar/dogs/others/dog4"),
    }


def test_datachain_save_with_job(test_session, catalog, datachain_job_id):
    DataChain.from_values(value=["val1", "val2"], session=test_session).save("my-ds")

    dataset = catalog.get_dataset("my-ds")
    result_job_id = dataset.get_version(dataset.latest_version).job_id
    assert result_job_id == datachain_job_id
