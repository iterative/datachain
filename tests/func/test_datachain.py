import os
import re

import pandas as pd
import pytest
from PIL import Image

from datachain.lib.dc import DataChain
from datachain.lib.file import File, ImageFile
from tests.utils import images_equal


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


@pytest.mark.parametrize("use_cache", [True, False])
def test_export_images_files(tmp_dir, tmp_path, use_cache):
    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    for img in images:
        img["data"].save(tmp_path / img["name"])

    DataChain.from_values(
        file=[
            ImageFile(name=img["name"], source=f"file://{tmp_path}") for img in images
        ],
    ).export_files(tmp_dir / "output", placement="filename", use_cache=use_cache)

    for img in images:
        exported_img = Image.open(tmp_dir / "output" / img["name"])
        assert images_equal(img["data"], exported_img)


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
