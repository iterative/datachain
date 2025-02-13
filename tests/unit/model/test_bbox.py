import numpy as np
import pytest

from datachain.model import BBox, OBBox


@pytest.mark.parametrize(
    "coords,title,normalized_to",
    [
        ((10, 20, 90, 80), "BBox", None),
        ([10, 20, 90, 80], "BBox", None),
        ([10.4, 19.8, 90.0, 80.1], "BBox", None),
        ([0.1, 0.2, 0.9, 0.8], "", (100, 100)),
    ],
)
def test_bbox_voc(coords, title, normalized_to):
    bbox = BBox.from_voc(
        coords,
        title=title,
        normalized_to=normalized_to,
    )
    assert bbox.model_dump() == {
        "title": title,
        "coords": [10, 20, 90, 80],
    }
    assert bbox.to_voc() == [10, 20, 90, 80]
    np.testing.assert_array_almost_equal(
        bbox.to_voc_normalized((100, 100)),
        [0.1, 0.2, 0.9, 0.8],
    )
    np.testing.assert_array_almost_equal(
        bbox.to_voc_normalized((200, 200)),
        [0.05, 0.1, 0.45, 0.4],
    )


@pytest.mark.parametrize(
    "coords,title,normalized_to",
    [
        ((10, 20, 80, 60), "BBox", None),
        ([10, 20, 80, 60], "BBox", None),
        ([9.9, 20.1, 80.4, 60.001], "BBox", None),
        ([0.1, 0.2, 0.8, 0.6], "", (100, 100)),
    ],
)
def test_bbox_coco(coords, title, normalized_to):
    bbox = BBox.from_coco(
        coords,
        title=title,
        normalized_to=normalized_to,
    )
    assert bbox.model_dump() == {
        "title": title,
        "coords": [10, 20, 90, 80],
    }
    assert bbox.to_coco() == [10, 20, 80, 60]
    np.testing.assert_array_almost_equal(
        bbox.to_coco_normalized((100, 100)),
        [0.1, 0.2, 0.8, 0.6],
    )
    np.testing.assert_array_almost_equal(
        bbox.to_coco_normalized((200, 200)),
        [0.05, 0.1, 0.4, 0.3],
    )


@pytest.mark.parametrize(
    "coords,title,normalized_to",
    [
        ((50, 50, 80, 60), "BBox", None),
        ([50, 50, 80, 60], "BBox", None),
        ([50.0, 49.6, 79.99, 60.2], "BBox", None),
        ([0.5, 0.5, 0.8, 0.6], "", (100, 100)),
    ],
)
def test_bbox_yolo(coords, title, normalized_to):
    bbox = BBox.from_yolo(
        coords,
        title=title,
        normalized_to=normalized_to,
    )
    assert bbox.model_dump() == {
        "title": title,
        "coords": [10, 20, 90, 80],
    }
    assert bbox.to_yolo() == [50, 50, 80, 60]
    np.testing.assert_array_almost_equal(
        bbox.to_yolo_normalized((100, 100)),
        [0.5, 0.5, 0.8, 0.6],
    )
    np.testing.assert_array_almost_equal(
        bbox.to_yolo_normalized((200, 200)),
        [0.25, 0.25, 0.4, 0.3],
    )


def test_bbox_from_list():
    assert BBox.from_list([10, 20, 90, 80]).model_dump() == {
        "title": "",
        "coords": [10, 20, 90, 80],
    }


def test_bbox_from_dict():
    assert BBox.from_dict({"x1": 10, "y1": 20, "x2": 90, "y2": 80}).model_dump() == {
        "title": "",
        "coords": [10, 20, 90, 80],
    }


@pytest.mark.parametrize(
    "coords",
    [
        {"x1": 10, "y1": 20, "x2": 90},
        {"x1": 10, "y1": 20, "x2": 90, "y2": 80, "x3": 100},
    ],
)
def test_bbox_from_dict_errors(coords):
    with pytest.raises(AssertionError):
        BBox.from_dict(coords)


@pytest.mark.parametrize(
    "coords,normalized_to",
    [
        [(10, 20, 30, 40, 50, 60, 70, 80), None],
        [[10, 20, 30, 40, 50, 60, 70, 80], None],
        [[9.9, 20.1, 29.6, 40.4, 50.01, 59.99, 70.0, 80], None],
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], None],
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], (100, 100)],
        [(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), [100, 100]],
    ],
)
def test_obbox_from_list(coords, normalized_to):
    obbox = OBBox.from_list(coords, normalized_to=normalized_to)
    assert obbox.model_dump() == {
        "title": "",
        "coords": [10, 20, 30, 40, 50, 60, 70, 80],
    }
    np.testing.assert_array_almost_equal(
        obbox.to_normalized((100, 100)),
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )


def test_obbox_to_normalized_errors():
    with pytest.raises(AssertionError):
        OBBox.from_list([10, 20, 30, 40, 50, 60, 70, 80]).to_normalized((50, 50))


@pytest.mark.parametrize(
    "coords,normalized_to",
    [
        [None, None],
        [12, None],
        ["12", None],
        [[], None],
        [[10, 20, 30, 40, 50, 60, 70], None],
        [[10, 20, 30, 40, 50, 60, 70, 80, 90], None],
        [[10, 20, 30, 40, 50, 60, 70, 80], (100, 100)],
        [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], (100, 100)],
    ],
)
def test_obbox_from_list_errors(coords, normalized_to):
    with pytest.raises(AssertionError):
        OBBox.from_list(coords, normalized_to=normalized_to)


def test_obbox_from_dict():
    obbox = OBBox.from_dict(
        {
            "x1": 0,
            "y1": 0.8,
            "x2": 2.2,
            "y2": 2.9,
            "x3": 3.9,
            "y3": 5.4,
            "x4": 6.0,
            "y4": 7.4,
        },
        title="OBBox",
    )
    assert obbox.model_dump() == {
        "title": "OBBox",
        "coords": [0, 1, 2, 3, 4, 5, 6, 7],
    }


@pytest.mark.parametrize(
    "coords",
    [
        {"x1": 0, "y1": 1, "x2": 2, "y2": 3, "x3": 4, "y3": 5, "x4": 6},
        {
            "x1": 0,
            "y1": 1,
            "x2": 2,
            "y2": 3,
            "x3": 4,
            "y3": 5,
            "x4": 6,
            "y4": 7,
            "x5": 8,
        },
    ],
)
def test_obbox_from_dict_errors(coords):
    with pytest.raises(AssertionError):
        OBBox.from_dict(coords)
