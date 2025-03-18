from collections.abc import Sequence
from typing import Literal

BBoxType = Literal["albumentations", "coco", "voc", "yolo"]


def validate_bbox(coords: Sequence[float], *types: type) -> None:
    """Validate the bounding box coordinates."""
    if not isinstance(coords, (list, tuple)):
        raise TypeError(
            f"Invalid bounding box coordinates: {coords}, should be a list or tuple"
        )
    if len(coords) != 4:
        raise ValueError(
            f"Invalid bounding box coordinates: {coords}, should have 4 values"
        )
    if any(not isinstance(c, types) for c in coords):
        raise ValueError(
            f"Invalid bounding box coordinates: {coords}, should be {types}"
        )


def convert_bbox(
    coords: Sequence[float],
    img_size: Sequence[int],
    source: BBoxType,
    target: BBoxType,
) -> list[float]:
    """
    Convert the bounding box coordinates between different formats.

    Supported formats: "albumentations", "coco", "voc", "yolo".

    Albumentations format represents bounding boxes as [x_min, y_min, x_max, y_max],
    where:
        - (x_min, y_min) are the normalized coordinates of the top-left corner.
        - (x_max, y_max) are the normalized coordinates of the bottom-right corner.

    COCO format represents bounding boxes as [x_min, y_min, width, height], where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - width and height define the size of the bounding box in pixels.

    PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max], where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - (x_max, y_max) are the pixel coordinates of the bottom-right corner.

    YOLO format represents bounding boxes as [x_center, y_center, width, height], where:
        - (x_center, y_center) are the normalized coordinates of the box center.
        - width and height normalized values define the size of the bounding box.

    Normalized coordinates are floats between 0 and 1, representing the
    relative position of the pixels in the image.

    Args:
        coords (Sequence[float]): The bounding box coordinates to convert.
        img_size (Sequence[int]): The reference image size (width, height).
        source (str): The source bounding box format.
        target (str): The target bounding box format.

    Returns:
        list[float]: The bounding box coordinates in the target format.
    """
    if source == "albumentations":
        return [
            round(c, 4) for c in convert_albumentations_bbox(coords, img_size, target)
        ]
    if source == "coco":
        return [round(c, 4) for c in convert_coco_bbox(coords, img_size, target)]
    if source == "voc":
        return [round(c, 4) for c in convert_voc_bbox(coords, img_size, target)]
    if source == "yolo":
        return [round(c, 4) for c in convert_yolo_bbox(coords, img_size, target)]
    raise ValueError(f"Unsupported source format: {source}")


def convert_albumentations_bbox(
    coords: Sequence[float],
    img_size: Sequence[int],
    target: BBoxType,
) -> list[float]:
    """Convert the Albumentations bounding box coordinates to other formats."""
    if target == "albumentations":
        return list(coords)
    if target == "coco":
        return [
            coords[0] * img_size[0],
            coords[1] * img_size[1],
            (coords[2] - coords[0]) * img_size[0],
            (coords[3] - coords[1]) * img_size[1],
        ]
    if target == "voc":
        return [coords[i] * img_size[i % 2] for i in range(4)]
    if target == "yolo":
        return [
            (coords[0] + coords[2]) / 2,
            (coords[1] + coords[3]) / 2,
            coords[2] - coords[0],
            coords[3] - coords[1],
        ]
    raise ValueError(f"Unsupported target format: {target}")


def convert_coco_bbox(
    coords: Sequence[float],
    img_size: Sequence[int],
    target: BBoxType,
) -> list[float]:
    """Convert the COCO bounding box coordinates to other formats."""
    if target == "albumentations":
        return [
            coords[0] / img_size[0],
            coords[1] / img_size[1],
            (coords[0] + coords[2]) / img_size[0],
            (coords[1] + coords[3]) / img_size[1],
        ]
    if target == "coco":
        return list(coords)
    if target == "voc":
        return [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
    if target == "yolo":
        return [
            (coords[0] + coords[2] / 2) / img_size[0],
            (coords[1] + coords[3] / 2) / img_size[1],
            coords[2] / img_size[0],
            coords[3] / img_size[1],
        ]
    raise ValueError(f"Unsupported target format: {target}")


def convert_voc_bbox(
    coords: Sequence[float],
    img_size: Sequence[int],
    target: BBoxType,
) -> list[float]:
    """Convert the PASCAL VOC bounding box coordinates to other formats."""
    if target == "albumentations":
        return [
            coords[0] / img_size[0],
            coords[1] / img_size[1],
            coords[2] / img_size[0],
            coords[3] / img_size[1],
        ]
    if target == "coco":
        return [
            coords[0],
            coords[1],
            coords[2] - coords[0],
            coords[3] - coords[1],
        ]
    if target == "voc":
        return list(coords)
    if target == "yolo":
        return [
            (coords[0] + coords[2]) / 2 / img_size[0],
            (coords[1] + coords[3]) / 2 / img_size[1],
            (coords[2] - coords[0]) / img_size[0],
            (coords[3] - coords[1]) / img_size[1],
        ]
    raise ValueError(f"Unsupported target format: {target}")


def convert_yolo_bbox(
    coords: Sequence[float],
    img_size: Sequence[int],
    target: BBoxType,
) -> list[float]:
    """Convert the YOLO bounding box coordinates to other formats."""
    if target == "albumentations":
        return [
            coords[0] - coords[2] / 2,
            coords[1] - coords[3] / 2,
            coords[0] + coords[2] / 2,
            coords[1] + coords[3] / 2,
        ]
    if target == "coco":
        return [
            (coords[0] - coords[2] / 2) * img_size[0],
            (coords[1] - coords[3] / 2) * img_size[1],
            coords[2] * img_size[0],
            coords[3] * img_size[1],
        ]
    if target == "voc":
        return [
            (coords[0] - coords[2] / 2) * img_size[0],
            (coords[1] - coords[3] / 2) * img_size[1],
            (coords[0] + coords[2] / 2) * img_size[0],
            (coords[1] + coords[3] / 2) * img_size[1],
        ]
    if target == "yolo":
        return list(coords)
    raise ValueError(f"Unsupported target format: {target}")
