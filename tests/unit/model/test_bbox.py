from numpy.testing import assert_array_almost_equal

from datachain.model import BBox, OBBox


def test_bbox():
    bbox = BBox(title="Object", coords=[10, 20, 90, 80])
    assert bbox.model_dump() == {
        "title": "Object",
        "coords": [10.0, 20.0, 90.0, 80.0],
    }


def test_bbox_convert():
    bbox = BBox(title="Object", coords=[10, 20, 90, 80])
    assert_array_almost_equal(
        bbox.convert([100, 100], "voc", "yolo"),
        [0.5, 0.5, 0.8, 0.6],
        decimal=3,
    )


def test_bbox_empty():
    assert BBox().model_dump() == {"title": "", "coords": []}


def test_obbox():
    obbox = OBBox(title="Object", coords=[10, 20, 30, 40, 50, 60, 70, 80])
    assert obbox.model_dump() == {
        "title": "Object",
        "coords": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    }


def test_obbox_empty():
    assert OBBox().model_dump() == {"title": "", "coords": []}
