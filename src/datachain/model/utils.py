from collections.abc import Sequence


def validate_img_size(img_size: Sequence[int]) -> Sequence[int]:
    """Validate the image size."""
    assert isinstance(img_size, (tuple, list)), "Image size must be a tuple or list."
    assert len(img_size) == 2, "Image size must be a tuple or list of 2 integers."
    assert all(isinstance(value, int) for value in img_size), (
        "Image size must be integers."
    )
    assert all(value > 0 for value in img_size), "Image size must be positive integers."
    return img_size


def validate_bbox(coords: Sequence[float]) -> Sequence[float]:
    """Validate the bounding box coordinates."""
    assert isinstance(coords, (tuple, list)), "Bounding box must be a tuple or list."
    assert len(coords) == 4, "Bounding box must be a tuple or list of 4 coordinates."
    assert all(isinstance(value, (int, float)) for value in coords), (
        "Bounding box coordinates must be floats or integers."
    )
    assert all(value >= 0 for value in coords), (
        "Bounding box coordinates must be positive."
    )
    return coords


def validate_bbox_normalized(
    coords: Sequence[float], img_size: Sequence[int]
) -> Sequence[float]:
    """Validate the bounding box coordinates and normalize them to the image size."""
    assert isinstance(coords, (tuple, list)), "Bounding box must be a tuple or list."
    assert len(coords) == 4, "Bounding box must be a tuple or list of 4 coordinates."
    assert all(isinstance(value, float) for value in coords), (
        "Bounding box normalized coordinates must be floats."
    )
    assert all(0 <= value <= 1 for value in coords), (
        "Bounding box normalized coordinates must be floats between 0 and 1."
    )

    width, height = validate_img_size(img_size)

    return [
        coords[0] * width,
        coords[1] * height,
        coords[2] * width,
        coords[3] * height,
    ]


def normalize_coords(
    coords: Sequence[int],
    img_size: Sequence[int],
) -> list[float]:
    """Normalize the bounding box coordinates to the image size."""
    assert isinstance(coords, (tuple, list)), "Coords must be a tuple or list."
    assert len(coords) == 4, "Coords must be a tuple or list of 4 coordinates."
    assert all(isinstance(value, int) for value in coords), (
        "Coords must be a tuple or list of 4 ints."
    )

    width, height = validate_img_size(img_size)

    assert (
        0 <= coords[0] <= width
        and 0 <= coords[1] <= height
        and 0 <= coords[2] <= width
        and 0 <= coords[3] <= height
    ), "Bounding box coordinates are out of image size"

    return [
        coords[0] / width,
        coords[1] / height,
        coords[2] / width,
        coords[3] / height,
    ]
