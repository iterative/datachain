from collections.abc import Sequence
from typing import Optional

from pydantic import Field

from datachain.lib.data_model import DataModel

from .utils import (
    normalize_coords,
    validate_bbox,
    validate_bbox_normalized,
    validate_img_size,
)


class BBox(DataModel):
    """
    A data model for representing bounding box.

    Attributes:
        title (str): The title of the bounding box.
        coords (list[int]): The coordinates of the bounding box.

    The bounding box is defined by two points with pixel coordinates:
        - (x1, y1): The top-left corner of the box.
        - (x2, y2): The bottom-right corner of the box.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_voc(
        coords: Sequence[float],
        title: str = "",
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "BBox":
        """
        Create a bounding box from coordinates in PASCAL VOC format.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
            - (x_min, y_min) are the coordinates of the top-left corner.
            - (x_max, y_max) are the coordinates of the bottom-right corner.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            coords (Sequence[float]): The bounding box coordinates.
            title (str, optional): The title or label for the bounding box.
                Defaults to "".
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            BBox: A bounding box object.
        """
        coords = (
            validate_bbox(coords)
            if normalized_to is None
            else validate_bbox_normalized(coords, normalized_to)
        )

        return BBox(title=title, coords=[round(c) for c in coords])

    def to_voc(self) -> list[int]:
        """
        Convert the bounding box to PASCAL VOC format.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
            - (x_min, y_min) are the coordinates of the top-left corner.
            - (x_max, y_max) are the coordinates of the bottom-right corner.

        Returns:
            list[int]: The bounding box coordinates in PASCAL VOC format.
        """
        return self.coords

    def to_voc_normalized(self, img_size: Sequence[int]) -> list[float]:
        """
        Convert the bounding box to PASCAL VOC format with normalized coordinates.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
            - (x_min, y_min) are the coordinates of the top-left corner.
            - (x_max, y_max) are the coordinates of the bottom-right corner.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            list[float]: The bounding box coordinates in PASCAL VOC format
                with normalized coordinates.
        """
        return normalize_coords(self.coords, img_size)

    @staticmethod
    def from_coco(
        coords: Sequence[float],
        title: str = "",
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "BBox":
        """
        Create a bounding box from coordinates in COCO format.

        COCO format represents bounding boxes as [x, y, width, height], where:
            - (x, y) are the coordinates of the top-left corner.
            - width and height define the size of the bounding box.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            coords (Sequence[float]): The bounding box coordinates.
            title (str, optional): The title or label for the bounding box.
                Defaults to "".
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            BBox: A bounding box object.
        """
        coords = (
            validate_bbox(coords)
            if normalized_to is None
            else validate_bbox_normalized(coords, normalized_to)
        )

        x, y, width, height = coords
        return BBox(
            title=title,
            coords=[round(x), round(y), round(x + width), round(y + height)],
        )

    def to_coco(self) -> list[int]:
        """
        Convert the bounding box to COCO format.

        COCO format represents bounding boxes as [x, y, width, height], where:
            - (x, y) are the coordinates of the top-left corner.
            - width and height define the size of the bounding box.

        Returns:
            list[int]: The bounding box coordinates in PASCAL VOC format.
        """
        return [
            self.coords[0],
            self.coords[1],
            self.coords[2] - self.coords[0],
            self.coords[3] - self.coords[1],
        ]

    def to_coco_normalized(self, img_size: Sequence[int]) -> list[float]:
        """
        Convert the bounding box to COCO format with normalized coordinates.

        COCO format represents bounding boxes as [x, y, width, height], where:
            - (x, y) are the coordinates of the top-left corner.
            - width and height define the size of the bounding box.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            list[float]: The bounding box coordinates in PASCAL VOC format
                with normalized coordinates.
        """
        coords_normalized = normalize_coords(self.coords, img_size)
        return [
            coords_normalized[0],
            coords_normalized[1],
            coords_normalized[2] - coords_normalized[0],
            coords_normalized[3] - coords_normalized[1],
        ]

    @staticmethod
    def from_yolo(
        coords: Sequence[float],
        title: str = "",
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "BBox":
        """
        Create a bounding box from coordinates in YOLO format.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
            - (x_center, y_center) are the coordinates of the box center.
            - width and height define the size of the bounding box.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            coords (Sequence[float]): The bounding box coordinates.
            title (str, optional): The title or label for the bounding box.
                Defaults to "".
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            BBox: The bounding box object.
        """
        coords = (
            validate_bbox(coords)
            if normalized_to is None
            else validate_bbox_normalized(coords, normalized_to)
        )

        x_center, y_center, width, height = coords
        return BBox(
            title=title,
            coords=[
                round(x_center - width / 2),
                round(y_center - height / 2),
                round(x_center + width / 2),
                round(y_center + height / 2),
            ],
        )

    def to_yolo(self) -> list[int]:
        """
        Convert the bounding box to YOLO format.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
            - (x_center, y_center) are the coordinates of the box center.
            - width and height define the size of the bounding box.

        Returns:
            list[int]: The bounding box coordinates in PASCAL VOC format.
        """
        return [
            round((self.coords[0] + self.coords[2]) / 2),
            round((self.coords[1] + self.coords[3]) / 2),
            self.coords[2] - self.coords[0],
            self.coords[3] - self.coords[1],
        ]

    def to_yolo_normalized(self, img_size: Sequence[int]) -> list[float]:
        """
        Convert the bounding box to YOLO format with normalized coordinates.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
            - (x_center, y_center) are the coordinates of the box center.
            - width and height define the size of the bounding box.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            list[float]: The bounding box coordinates in PASCAL VOC format
                with normalized coordinates.
        """
        coords_normalized = normalize_coords(self.coords, img_size)
        return [
            (coords_normalized[0] + coords_normalized[2]) / 2,
            (coords_normalized[1] + coords_normalized[3]) / 2,
            coords_normalized[2] - coords_normalized[0],
            coords_normalized[3] - coords_normalized[1],
        ]

    @staticmethod
    def from_list(coords: Sequence[float], title: str = "") -> "BBox":
        return BBox.from_voc(coords, title)

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "BBox":
        keys = ("x1", "y1", "x2", "y2")
        assert isinstance(coords, dict) and set(coords) == set(keys), (
            "Bounding box must be a dictionary with keys 'x1', 'y1', 'x2' and 'y2'."
        )
        return BBox.from_voc([coords[key] for key in keys], title=title)


class OBBox(DataModel):
    """
    A data model for representing oriented bounding boxes.

    Attributes:
        title (str): The title of the oriented bounding box.
        coords (list[int]): The coordinates of the oriented bounding box.

    The oriented bounding box is defined by four points:
        - (x1, y1): The first corner of the box.
        - (x2, y2): The second corner of the box.
        - (x3, y3): The third corner of the box.
        - (x4, y4): The fourth corner of the box.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_list(
        coords: Sequence[float],
        title: str = "",
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "OBBox":
        """
        Create an oriented bounding box from a list of coordinates.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            coords (Sequence[float]): The oriented bounding box coordinates.
            title (str, optional): The title or label for the oriented bounding box.
                Defaults to "".
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the oriented bounding box.
                If None (default), the coordinates are assumed to be
                absolute pixel values.

        Returns:
            OBBox: An oriented bounding box object.
        """
        assert isinstance(coords, (tuple, list)), (
            "Oriented bounding box must be a tuple or list."
        )
        assert len(coords) == 8, (
            "Oriented bounding box must be a tuple or list of 8 coordinates."
        )
        assert all(isinstance(value, (int, float)) for value in coords), (
            "Oriented bounding box coordinates must be floats or integers."
        )

        if normalized_to is not None:
            assert all(0 <= coord <= 1 for coord in coords), (
                "Normalized coordinates must be floats between 0 and 1."
            )
            width, height = validate_img_size(normalized_to)
            coords = [
                coord * width if i % 2 == 0 else coord * height
                for i, coord in enumerate(coords)
            ]

        return OBBox(
            title=title,
            coords=[round(c) for c in coords],
        )

    def to_normalized(self, img_size: Sequence[int]) -> list[float]:
        """
        Return the oriented bounding box in normalized coordinates.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            list[float]: The oriented bounding box with normalized coordinates.
        """
        width, height = validate_img_size(img_size)
        assert all(
            x < width and y < height
            for x, y in zip(self.coords[::2], self.coords[1::2])
        ), "Oriented bounding box is out of image size."
        return [
            coord / width if i % 2 == 0 else coord / height
            for i, coord in enumerate(self.coords)
        ]

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "OBBox":
        keys = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
        assert isinstance(coords, dict) and set(coords) == set(keys), (
            "Oriented bounding box must be a dictionary with coordinates."
        )
        return OBBox.from_list([coords[key] for key in keys], title=title)
