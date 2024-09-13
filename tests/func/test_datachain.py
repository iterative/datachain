import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import pytz
from PIL import Image
from sqlalchemy import Column

from datachain.data_storage.sqlite import SQLiteWarehouse
from datachain.dataset import DatasetStats
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
from tests.utils import TARRED_TREE, images_equal


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
    assert chain.catalog.client_config.get("anon", False) is anon


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
        print(r)
        count += 1
        assert len(r[0]) == r[1]
    assert count == 7


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


@pytest.mark.parametrize("tree", [TARRED_TREE], indirect=True)
def test_process_and_open_tar(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = (
        DataChain.from_storage(ctc.src_uri, session=ctc.session)
        .gen(file=process_tar)
        .filter(C("file.path").glob("*/cats/*"))
    )
    assert dc.count() == 2
    assert {(file.read(), file.name) for file in dc.collect("file")} == {
        (b"meow", "cat1"),
        (b"mrow", "cat2"),
    }
