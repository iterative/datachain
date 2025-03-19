import pytest
from numpy.testing import assert_array_almost_equal

from datachain.model.utils import convert_bbox, validate_bbox

# Test data: list of bounding boxes in different formats.
# These meant to be the same bounding boxes assuming image size is (100, 100).
# Formats are:
# - albumentations: [x_min, y_min, x_max, y_max], normalized coordinates
# - coco: [x_min, y_min, width, height], pixel coordinates
# - voc: [x_min, y_min, x_max, y_max], pixel coordinates
# - yolo: [x_center, y_center, width, height], normalized coordinates
BOXES = [
    {
        "albumentations": [0.0, 0.0, 0.0, 0.0],
        "coco": [0, 0, 0, 0],
        "voc": [0, 0, 0, 0],
        "yolo": [0.0, 0.0, 0.0, 0.0],
    },
    {
        "albumentations": [0.5, 0.5, 0.5, 0.5],
        "coco": [50, 50, 0, 0],
        "voc": [50, 50, 50, 50],
        "yolo": [0.5, 0.5, 0.0, 0.0],
    },
    {
        "albumentations": [1.0, 1.0, 1.0, 1.0],
        "coco": [100, 100, 0, 0],
        "voc": [100, 100, 100, 100],
        "yolo": [1.0, 1.0, 0.0, 0.0],
    },
    {
        "albumentations": [0.0, 0.0, 1.0, 1.0],
        "coco": [0, 0, 100, 100],
        "voc": [0, 0, 100, 100],
        "yolo": [0.5, 0.5, 1.0, 1.0],
    },
    {
        "albumentations": [0.1, 0.2, 0.9, 0.8],
        "coco": [10, 20, 80, 60],
        "voc": [10, 20, 90, 80],
        "yolo": [0.5, 0.5, 0.8, 0.6],
    },
]


@pytest.mark.parametrize(
    "coords,types,exception",
    [
        (None, [int], TypeError),
        ([], [int], ValueError),
        ([10, 20, 90], [int], ValueError),
        ([10, 20, 90, 80, 100], [int], ValueError),
        ([10, 20, "90", 80], [int], ValueError),
        ([10, 20, 90, 80], [float], ValueError),
    ],
)
def test_validate_bbox(coords, types, exception):
    with pytest.raises(exception):
        validate_bbox(coords, *types)


@pytest.mark.parametrize(
    "source,target,coords,result",
    [
        (source, target, coords, result)
        for box in BOXES
        for source, coords in box.items()
        for target, result in box.items()
    ],
)
def test_convert_bbox(source, target, coords, result):
    assert_array_almost_equal(
        convert_bbox(coords, (100, 100), source, target),
        result,
        decimal=3,
    )


@pytest.mark.parametrize(
    "source,target",
    [
        ("unknown", "coco"),
        ("albumentations", "unknown"),
        ("coco", "unknown"),
        ("voc", "unknown"),
        ("yolo", "unknown"),
    ],
)
def test_convert_bbox_error_source_target(source, target):
    with pytest.raises(ValueError):
        convert_bbox([0, 0, 0, 0], (100, 100), source, target)
