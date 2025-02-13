import pytest

from datachain.model.utils import (
    normalize_coords,
    validate_bbox,
    validate_bbox_normalized,
    validate_img_size,
)


@pytest.mark.parametrize(
    "img_size",
    [
        [100, 100],
        (100, 100),
    ],
)
def test_validate_img_size(img_size):
    assert validate_img_size(img_size) == img_size


@pytest.mark.parametrize(
    "img_size",
    [
        None,
        12,
        "12",
        [],
        [1],
        [1, 2, 3],
        [1, "2"],
        [0, 2],
        [1, 0],
        [10.0, 10.0],
    ],
)
def test_validate_img_size_errors(img_size):
    with pytest.raises(AssertionError):
        validate_img_size(img_size)


@pytest.mark.parametrize(
    "bbox",
    [
        (10, 10, 90, 90),
        [10, 10, 90, 90],
    ],
)
def test_validate_bbox(bbox):
    assert validate_bbox(bbox) == bbox


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        12,
        "12",
        [],
        [0, 1, 2],
        [0, 1, 2, 3, 4],
        [0, 1, 2, "3"],
        [0, -1, 2, 3],
    ],
)
def test_validate_bbox_errors(bbox):
    with pytest.raises(AssertionError):
        validate_bbox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [
        (0.1, 0.1, 0.9, 0.9),
        [0.1, 0.1, 0.9, 0.9],
    ],
)
def test_validate_bbox_normalized(bbox):
    assert validate_bbox_normalized(bbox, (100, 100)) == [10, 10, 90, 90]


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        0.2,
        "0.2",
        [],
        [0.0, 0.1, 0.2],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [0.0, 0.1, 0.2, "0.3"],
        [0.0, 1.0, 2.0, 3.0],
    ],
)
def test_validate_bbox_normalized_errors(bbox):
    with pytest.raises(AssertionError):
        validate_bbox_normalized(bbox, (100, 100))


@pytest.mark.parametrize(
    "coords",
    [
        (10, 10, 90, 90),
        [10, 10, 90, 90],
    ],
)
def test_normalize_coords(coords):
    assert normalize_coords(coords, (100, 100)) == [0.1, 0.1, 0.9, 0.9]


@pytest.mark.parametrize(
    "coords",
    [
        None,
        10,
        "10",
        [10, 10, 90],
        [10, 10, 90, 90, 90],
        [10.0, 10.0, 90.0, 90.0],
        [200, 10, 90, 90],
        [10, 200, 90, 90],
        [10, 10, 200, 90],
        [10, 10, 90, 200],
        [-10, 10, 90, 90],
        [10, -10, 90, 90],
        [10, 10, -10, 90],
        [10, 10, 90, -10],
    ],
)
def test_normalize_coords_errors(coords):
    with pytest.raises(AssertionError):
        normalize_coords(coords, (100, 100))
