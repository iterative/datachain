import os

import pytest
import regex as re
from PIL import Image

import datachain as dc
from datachain import func
from datachain.error import DatasetVersionNotFoundError
from datachain.lib.dc import C
from datachain.lib.file import File, ImageFile


def _get_dependencies(catalog, name, version) -> list[tuple[str, str]]:
    return sorted(
        [
            (d.name, d.version)
            for d in catalog.get_dataset_dependencies(name, version, indirect=False)
        ]
    )


def test_delta_update_from_dataset(test_session, tmp_dir, tmp_path):
    catalog = test_session.catalog
    starting_ds_name = "starting_ds"
    ds_name = "delta_ds"

    images = [
        {"name": "img1.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img2.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
        {"name": "img3.jpg", "data": Image.new(mode="RGB", size=(64, 64))},
        {"name": "img4.jpg", "data": Image.new(mode="RGB", size=(128, 128))},
    ]

    def create_image_dataset(ds_name, images):
        dc.read_values(
            file=[
                ImageFile(path=img["name"], source=f"file://{tmp_path}")
                for img in images
            ],
            session=test_session,
        ).save(ds_name)

    def create_delta_dataset(ds_name):
        dc.read_dataset(
            starting_ds_name,
            session=test_session,
            delta=True,
            delta_on=["file.source", "file.path"],
            delta_result_on=["file.source", "file.path"],
            delta_compare=["file.version", "file.etag"],
        ).save(ds_name)

    # first version of starting dataset
    create_image_dataset(starting_ds_name, images[:2])
    # first version of delta dataset
    create_delta_dataset(ds_name)
    assert _get_dependencies(catalog, ds_name, "1.0.0") == [(starting_ds_name, "1.0.0")]
    # second version of starting dataset
    create_image_dataset(starting_ds_name, images[2:])
    # second version of delta dataset
    create_delta_dataset(ds_name)
    assert _get_dependencies(catalog, ds_name, "1.0.1") == [(starting_ds_name, "1.0.1")]

    assert list(
        dc.read_dataset(ds_name, version="1.0.0")
        .order_by("file.path")
        .collect("file.path")
    ) == [
        "img1.jpg",
        "img2.jpg",
    ]

    assert list(
        dc.read_dataset(ds_name, version="1.0.1")
        .order_by("file.path")
        .collect("file.path")
    ) == [
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
        "img4.jpg",
    ]

    create_delta_dataset(ds_name)


def test_delta_update_from_storage(test_session, tmp_dir, tmp_path):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)

    images = [
        {
            "name": f"img{i}.{'jpg' if i % 2 == 0 else 'png'}",
            "data": Image.new(mode="RGB", size=((i + 1) * 10, (i + 1) * 10)),
        }
        for i in range(20)
    ]

    # save only half of the images for now
    for img in images[:10]:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def my_embedding(file: File) -> list[float]:
            return [0.5, 0.5]

        def get_index(file: File) -> int:
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_result_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .filter(C("file.path").glob("*.jpg"))
            .map(emb=my_embedding)
            .mutate(dist=func.cosine_distance("emb", (0.1, 0.2)))
            .map(index=get_index)
            .filter(C("index") > 3)
            .save(ds_name)
        )

    # first version of delta dataset
    create_delta_dataset()

    # remember old etags for later comparison to prove modified images are also taken
    # into consideration on delta update
    etags = {
        r[0]: r[1].etag
        for r in dc.read_dataset(ds_name, version="1.0.0").collect("index", "file")
    }

    # remove last couple of images to simulate modification since we will re-create it
    for img in images[5:10]:
        os.remove(tmp_dir / img["name"])

    # save other half of images and the ones that are removed above
    for img in images[5:]:
        img["data"].save(tmp_dir / img["name"])

    # remove first 5 images to check that deleted rows are not taken into consideration
    for img in images[0:5]:
        os.remove(tmp_dir / img["name"])

    # second version of delta dataset
    create_delta_dataset()

    assert list(
        dc.read_dataset(ds_name, version="1.0.0")
        .order_by("file.path")
        .collect("file.path")
    ) == [
        "images/img4.jpg",
        "images/img6.jpg",
        "images/img8.jpg",
    ]

    assert list(
        dc.read_dataset(ds_name, version="1.0.1")
        .order_by("file.path")
        .collect("file.path")
    ) == [
        "images/img10.jpg",
        "images/img12.jpg",
        "images/img14.jpg",
        "images/img16.jpg",
        "images/img18.jpg",
        "images/img4.jpg",
        "images/img6.jpg",
        "images/img8.jpg",
    ]

    # check that we have newest versions for modified rows since etags are mtime
    # and modified rows etags should be bigger than the old ones
    assert (
        next(
            dc.read_dataset(ds_name, version="1.0.1")
            .filter(C("index") == 6)
            .order_by("file.path", "file.etag")
            .collect("file.etag")
        )
        > etags[6]
    )


def test_delta_update_check_num_calls(test_session, tmp_dir, tmp_path, capsys):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)
    map_print = "In map"

    images = [
        {
            "name": f"img{i}.jpg",
            "data": Image.new(mode="RGB", size=((i + 1) * 10, (i + 1) * 10)),
        }
        for i in range(20)
    ]

    # save only half of the images for now
    for img in images[:10]:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def get_index(file: File) -> int:
            print(map_print)  # needed to count number of map calls
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_result_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .map(index=get_index)
            .save(ds_name)
        )

    # first version of delta dataset
    create_delta_dataset()
    # save other half of images
    for img in images[10:]:
        img["data"].save(tmp_dir / img["name"])
    # second version of delta dataset
    create_delta_dataset()

    captured = capsys.readouterr()
    # assert captured.out == "Garbage collecting 2 tables.\n"
    assert captured.out == "\n".join([map_print] * 20) + "\n"


def test_delta_update_no_diff(test_session, tmp_dir, tmp_path):
    ds_name = "delta_ds"
    path = tmp_dir.as_uri()
    tmp_dir = tmp_dir / "images"
    os.mkdir(tmp_dir)

    images = [
        {"name": f"img{i}.jpg", "data": Image.new(mode="RGB", size=(64, 128))}
        for i in range(10)
    ]

    for img in images:
        img["data"].save(tmp_dir / img["name"])

    def create_delta_dataset():
        def get_index(file: File) -> int:
            r = r".+\/img(\d+)\.jpg"
            return int(re.search(r, file.path).group(1))  # type: ignore[union-attr]

        (
            dc.read_storage(
                path,
                update=True,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
                delta_compare=["file.version", "file.etag"],
            )
            .filter(C("file.path").glob("*.jpg"))
            .map(index=get_index)
            .filter(C("index") > 5)
            .save(ds_name)
        )

    create_delta_dataset()
    create_delta_dataset()

    assert list(
        dc.read_dataset(ds_name, version="1.0.0")
        .order_by("file.path")
        .collect("file.path")
    ) == [
        "images/img6.jpg",
        "images/img7.jpg",
        "images/img8.jpg",
        "images/img9.jpg",
    ]

    with pytest.raises(DatasetVersionNotFoundError) as exc_info:
        dc.read_dataset(ds_name, version="1.0.1")

    assert str(exc_info.value) == f"Dataset {ds_name} does not have version 1.0.1"


@pytest.fixture
def file_dataset(test_session):
    return dc.read_values(
        file=[
            File(path="a.jpg", source="s3://bucket"),
            File(path="b.jpg", source="s3://bucket"),
        ],
        session=test_session,
    ).save("file_ds")


def test_delta_update_union(test_session, file_dataset):
    dc.read_values(num=[10, 20], session=test_session).save("numbers")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).union(dc.read_dataset("numbers"), session=test_session)
        )

    assert str(excinfo.value) == "Delta update cannot be used with union"


def test_delta_update_merge(test_session, file_dataset):
    dc.read_values(num=[10, 20], session=test_session).save("numbers")

    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).merge(dc.read_dataset("numbers"), on="id", session=test_session)
        )

    assert str(excinfo.value) == "Delta update cannot be used with merge"


def test_delta_update_distinct(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).distinct("file.path")
        )

    assert str(excinfo.value) == "Delta update cannot be used with distinct"


def test_delta_update_group_by(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).group_by(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == "Delta update cannot be used with group_by"


def test_delta_update_agg(test_session, file_dataset):
    with pytest.raises(NotImplementedError) as excinfo:
        (
            dc.read_dataset(
                file_dataset.name,
                session=test_session,
                delta=True,
                delta_on=["file.source", "file.path"],
            ).agg(cnt=func.count(), partition_by="file.path")
        )

    assert str(excinfo.value) == "Delta update cannot be used with agg"
