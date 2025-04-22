import os

import pytest
import regex as re
from PIL import Image

import datachain as dc
from datachain import func
from datachain.lib.dc import C
from datachain.lib.file import File, ImageFile


def test_delta_update_from_dataset(test_session, tmp_dir, tmp_path):
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
        ).save(ds_name, delta=True)

    # first version of starting dataset
    create_image_dataset(starting_ds_name, images[:2])
    # first version of delta dataset
    create_delta_dataset(ds_name)
    # second version of starting dataset
    create_image_dataset(starting_ds_name, images[2:])
    # second version of delta dataset
    create_delta_dataset(ds_name)

    assert list(
        dc.read_dataset(ds_name, version=1).order_by("file.path").collect("file.path")
    ) == [
        "img1.jpg",
        "img2.jpg",
    ]

    assert list(
        dc.read_dataset(ds_name, version=2).order_by("file.path").collect("file.path")
    ) == [
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
        "img4.jpg",
    ]


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
            dc.read_storage(path, update=True, session=test_session)
            .filter(C("file.path").glob("*.jpg"))
            .map(emb=my_embedding)
            .mutate(dist=func.cosine_distance("emb", (0.1, 0.2)))
            .map(index=get_index)
            .filter(C("index") > 3)
            .save(ds_name, delta=True)
        )

    # first version of delta dataset
    create_delta_dataset()

    # remember old etags for later comparison to prove modified images are also taken
    # into consideration on delta update
    etags = {
        r[0]: r[1].etag
        for r in dc.read_dataset(ds_name, version=1).collect("index", "file")
    }

    # remove last couple of images to simulate modification since we will re-create it
    for img in images[5:10]:
        os.remove(tmp_dir / img["name"])

    # save other half of images and the ones that are removed above
    for img in images[5:]:
        img["data"].save(tmp_dir / img["name"])

    # second version of delta dataset
    create_delta_dataset()

    assert list(
        dc.read_dataset(ds_name, version=1).order_by("file.path").collect("file.path")
    ) == [
        "images/img4.jpg",
        "images/img6.jpg",
        "images/img8.jpg",
    ]

    assert list(
        dc.read_dataset(ds_name, version=2).order_by("file.path").collect("file.path")
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
            dc.read_dataset(ds_name, version=2)
            .filter(C("index") == 6)
            .order_by("file.path", "file.etag")
            .collect("file.etag")
        )
        > etags[6]
    )


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
            dc.read_storage(path, update=True, session=test_session)
            .filter(C("file.path").glob("*.jpg"))
            .map(index=get_index)
            .filter(C("index") > 5)
            .save(ds_name, delta=True)
        )

    create_delta_dataset()
    create_delta_dataset()

    assert (
        list(
            dc.read_dataset(ds_name, version=1)
            .order_by("file.path")
            .collect("file.path")
        )
        == list(
            dc.read_dataset(ds_name, version=2)
            .order_by("file.path")
            .collect("file.path")
        )
        == [
            "images/img6.jpg",
            "images/img7.jpg",
            "images/img8.jpg",
            "images/img9.jpg",
        ]
    )


def test_delta_update_no_file_signals(test_session):
    starting_ds_name = "starting_ds"

    dc.read_values(num=[10, 20], session=test_session).save(starting_ds_name)

    with pytest.raises(ValueError) as excinfo:
        dc.read_dataset(
            starting_ds_name,
            session=test_session,
        ).save("delta_ds", delta=True)

    assert (
        str(excinfo.value)
        == "Chain doesn't produce file signal, cannot do delta update"
    )
