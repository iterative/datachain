import pytest

from datachain.model import BBox, OBBox, Pose


def test_bbox():
    bbox = BBox(title="Object", coords=[10, 20, 90, 80])
    assert bbox.model_dump() == {
        "title": "Object",
        "coords": [10, 20, 90, 80],
    }


def test_bbox_default_values():
    assert BBox().model_dump() == {"title": "", "coords": []}


@pytest.mark.parametrize(
    "coords",
    [
        [0.1, 0.2, 0.9, 0.8],
        (0.1, 0.2, 0.9, 0.8),
    ],
)
def test_bbox_albumentations(coords):
    img_size = (100, 100)
    bbox = BBox.from_albumentations(coords, img_size)
    assert bbox.model_dump() == {"title": "", "coords": [10, 20, 90, 80]}
    assert bbox.to_albumentations(img_size) == [0.1, 0.2, 0.9, 0.8]


@pytest.mark.parametrize(
    "coords,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([10, 20, 90], ValueError),
        ([10, 20, 90, 80], ValueError),
        ([10, 20, 90, 80, 100], ValueError),
        ([10, 20, "90", 80], ValueError),
    ],
)
def test_bbox_from_albumentations_error(coords, exception):
    with pytest.raises(exception):
        BBox.from_albumentations(coords, (100, 100))


@pytest.mark.parametrize(
    "coords",
    [
        [10, 20, 80, 60],
        (10, 20, 80, 60),
        [10.1, 19.9, 80.4, 60.0],
    ],
)
def test_bbox_coco(coords):
    bbox = BBox.from_coco(coords)
    assert bbox.model_dump() == {"title": "", "coords": [10, 20, 90, 80]}
    assert bbox.to_coco() == [10, 20, 80, 60]


@pytest.mark.parametrize(
    "coords,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([10, 20, 90], ValueError),
        ([10, 20, 90, 80, 100], ValueError),
        ([10, 20, "90", 80], ValueError),
    ],
)
def test_bbox_from_coco_error(coords, exception):
    with pytest.raises(exception):
        BBox.from_coco(coords)


@pytest.mark.parametrize(
    "coords",
    [
        [10, 20, 90, 80],
        (10, 20, 90, 80),
        [10.1, 19.9, 90.4, 80.0],
    ],
)
def test_bbox_voc(coords):
    bbox = BBox.from_voc(coords)
    assert bbox.model_dump() == {"title": "", "coords": [10, 20, 90, 80]}
    assert bbox.to_voc() == [10, 20, 90, 80]


@pytest.mark.parametrize(
    "coords,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([10, 20, 90], ValueError),
        ([10, 20, 90, 80, 100], ValueError),
        ([10, 20, "90", 80], ValueError),
    ],
)
def test_bbox_from_voc_error(coords, exception):
    with pytest.raises(exception):
        BBox.from_voc(coords)


@pytest.mark.parametrize(
    "coords",
    [
        [0.5, 0.5, 0.8, 0.6],
        (0.5, 0.5, 0.8, 0.6),
    ],
)
def test_bbox_yolo(coords):
    img_size = (100, 100)
    bbox = BBox.from_yolo(coords, img_size)
    assert bbox.model_dump() == {"title": "", "coords": [10, 20, 90, 80]}
    assert bbox.to_yolo(img_size) == [0.5, 0.5, 0.8, 0.6]


@pytest.mark.parametrize(
    "coords,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([10, 20, 90], ValueError),
        ([10, 20, 90, 80], ValueError),
        ([10, 20, 90, 80, 100], ValueError),
        ([10, 20, "90", 80], ValueError),
    ],
)
def test_bbox_from_yolo_error(coords, exception):
    with pytest.raises(exception):
        BBox.from_yolo(coords, (100, 100))


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (0, 0, False),
        (10, 20, True),
        (90, 80, True),
        (90, 10, False),
        (10, 90, False),
        (50, 50, True),
        (100, 100, False),
    ],
)
def test_bbox_point_inside(x, y, expected):
    bbox = BBox(coords=[10, 20, 90, 80])
    assert bbox.point_inside(x, y) == expected


@pytest.mark.parametrize(
    "pose,expected",
    [
        (Pose(x=[], y=[]), True),
        (Pose(x=[10, 90, 50], y=[20, 80, 50]), True),
        (Pose(x=[9, 90, 50], y=[20, 80, 50]), False),
    ],
)
def test_bbox_pose_inside(pose, expected):
    bbox = BBox(coords=[10, 20, 90, 80])
    assert bbox.pose_inside(pose) == expected


@pytest.mark.parametrize(
    "coords",
    [
        [10, 20, 90, 80],
        (10, 20, 90, 80),
        [10.1, 19.9, 90.4, 80.0],
    ],
)
def test_bbox_from_list(coords):
    assert BBox.from_list(coords).model_dump() == BBox.from_voc(coords).model_dump()


def test_bbox_from_dict():
    bbox = BBox.from_dict({"x1": 10, "y1": 20, "x2": 90, "y2": 80}, title="Object")
    assert bbox.model_dump() == {
        "title": "Object",
        "coords": [10, 20, 90, 80],
    }


def test_bbox_from_dict_error():
    with pytest.raises(ValueError):
        BBox.from_dict({"x1": 10, "y1": 20, "x2": 90}, title="Object")


def test_obbox():
    obbox = OBBox(title="Object", coords=[10, 20, 30, 40, 50, 60, 70, 80])
    assert obbox.model_dump() == {
        "title": "Object",
        "coords": [10, 20, 30, 40, 50, 60, 70, 80],
    }


def test_obbox_default_values():
    assert OBBox().model_dump() == {"title": "", "coords": []}


def test_obbox_from_list():
    obbox = OBBox.from_list([10, 20, 30, 40, 50, 60, 70, 80])
    assert obbox.model_dump() == {
        "title": "",
        "coords": [10, 20, 30, 40, 50, 60, 70, 80],
    }


@pytest.mark.parametrize(
    "coords,exception",
    [
        (None, TypeError),
        ([], ValueError),
        ([10, 20, 30, 40, 50, 60, 70], ValueError),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90], ValueError),
        ([10, 20, 30, 40, 50, 60, "70", 80], ValueError),
    ],
)
def test_obbox_from_list_error(coords, exception):
    with pytest.raises(exception):
        OBBox.from_list(coords)


def test_obbox_from_dict():
    obbox = OBBox.from_dict(
        {
            "x1": 10,
            "y1": 20,
            "x2": 30,
            "y2": 40,
            "x3": 50,
            "y3": 60,
            "x4": 70,
            "y4": 80,
        },
        title="Object",
    )
    assert obbox.model_dump() == {
        "title": "Object",
        "coords": [10, 20, 30, 40, 50, 60, 70, 80],
    }


def test_obbox_from_dict_error():
    with pytest.raises(ValueError):
        OBBox.from_dict(
            {
                "x1": 10,
                "y1": 20,
                "x2": 30,
                "y2": 40,
                "x3": 50,
                "y3": 60,
                "x4": 70,
            },
            title="Object",
        )
