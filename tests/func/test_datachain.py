import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from datachain.dataset import DatasetStats
from datachain.lib.dc import DataChain
from datachain.lib.file import File


@pytest.mark.parametrize("anon", [True, False])
def test_catalog_anon(catalog, anon):
    chain = (
        DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/", anon=anon)
        .limit(5)
        .save("test_catalog_anon")
    )
    assert chain.catalog.client_config.get("anon", False) is anon


def test_from_storage(cloud_test_catalog):
    ctc = cloud_test_catalog
    dc = DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
    assert dc.count() == 7


def test_from_storage_reindex(tmp_dir, catalog):
    path = tmp_dir.as_uri()

    pd.DataFrame({"name": ["Alice", "Bob"]}).to_parquet(tmp_dir / "test1.parquet")
    assert DataChain.from_storage(path, catalog=catalog).count() == 1

    pd.DataFrame({"name": ["Charlie", "David"]}).to_parquet(tmp_dir / "test2.parquet")
    assert DataChain.from_storage(path, catalog=catalog).count() == 1
    assert DataChain.from_storage(path, catalog=catalog, update=True).count() == 2


@pytest.mark.parametrize("use_cache", [False, True])
def test_map_file(cloud_test_catalog, use_cache):
    ctc = cloud_test_catalog

    def new_signal(file: File) -> str:
        with file.open() as f:
            return file.name + " -> " + f.read().decode("utf-8")

    dc = (
        DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
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

    dc = DataChain.from_storage(ctc.src_uri, catalog=ctc.catalog)
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
    tmp_dir, cloud_test_catalog, placement, use_map, use_cache, file_type
):
    ctc = cloud_test_catalog
    df = DataChain.from_storage(ctc.src_uri, type=file_type)
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


def test_export_files_filename_placement_not_unique_files(tmp_dir, catalog):
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

    df = DataChain.from_storage((tmp_dir / bucket_name).as_uri())
    with pytest.raises(ValueError):
        df.export_files(tmp_dir / "output", placement="filename")


def test_show(capsys, catalog):
    first_name = ["Alice", "Bob", "Charlie"]
    DataChain.from_values(
        first_name=first_name,
        age=[40, 30, None],
        city=[
            "Houston",
            "Los Angeles",
            None,
        ],
    ).show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert "first_name age city" in normalized_output
    for i in range(3):
        assert f"{i} {first_name[i]}" in normalized_output


def test_show_transpose(capsys, catalog):
    first_name = ["Alice", "Bob", "Charlie"]
    last_name = ["A", "B", "C"]
    DataChain.from_values(
        first_name=first_name,
        last_name=last_name,
    ).show(transpose=True)
    captured = capsys.readouterr()
    stripped_output = re.sub(r"\s+", " ", captured.out)
    assert " ".join(first_name) in stripped_output
    assert " ".join(last_name) in stripped_output


def test_show_truncate(capsys, catalog):
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
    )

    dc.show()
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    assert f"{client[0]} {details[0][:10]}" in normalized_output
    assert details[0] not in normalized_output
    for i in [1, 2]:
        assert f"{client[i]} {details[i]}" in normalized_output


def test_show_no_truncate(capsys, catalog):
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
    )

    dc.show(truncate=False)
    captured = capsys.readouterr()
    normalized_output = re.sub(r"\s+", " ", captured.out)
    for i in range(3):
        assert client[i] in normalized_output
        assert details[i] in normalized_output


def test_from_storage_dataset_stats(tmp_dir, catalog):
    for i in range(4):
        (tmp_dir / f"file{i}.txt").write_text(f"file{i}")

    dc = DataChain.from_storage(tmp_dir.as_uri(), catalog=catalog).save("test-data")
    stats = catalog.dataset_stats(dc.name, dc.version)
    assert stats == DatasetStats(num_objects=4, size=20)


def test_from_storage_check_rows(tmp_dir, catalog):
    stats = {}
    for i in range(4):
        file = tmp_dir / f"{i}.txt"
        file.write_text(f"file{i}")
        stats[file.name] = file.stat()

    dc = DataChain.from_storage(tmp_dir.as_uri(), catalog=catalog).save("test-data")

    for (file,) in dc.collect():
        assert isinstance(file, File)
        stat = stats[file.name]
        assert file == File(
            source=Path(tmp_dir.anchor).as_uri(),
            parent=tmp_dir.relative_to(tmp_dir.anchor),
            name=file.name,
            size=stat.st_size,
            version="",
            etag=stat.st_mtime.hex(),
            is_latest=True,
            last_modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            location=None,
            vtype="",
        )
