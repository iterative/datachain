from collections.abc import Sequence
from typing import Optional

from pydantic import Field

from datachain.lib.data_model import DataModel

from .utils import validate_img_size


class Pose(DataModel):
    """
    A data model for representing pose keypoints.

    Attributes:
        x (list[int]): The x-coordinates of the keypoints.
        y (list[int]): The y-coordinates of the keypoints.

    The keypoints are represented as lists of x and y coordinates, where each index
    corresponds to a specific body part.
    """

    x: list[int] = Field(default=[])
    y: list[int] = Field(default=[])

    @staticmethod
    def from_list(
        points: Sequence[Sequence[float]],
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "Pose":
        """
        Create a Pose instance from a list of x and y coordinates.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            points (Sequence[Sequence[float]]): The x and y coordinates
                of the keypoints. List of 2 lists: x and y coordinates.
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            Pose: A Pose object.
        """
        assert isinstance(points, (tuple, list)), "Pose must be a list of 2 lists."
        assert len(points) == 2, "Pose must be a list of 2 lists: x and y coordinates."
        points_x, points_y = points
        assert isinstance(points_x, (tuple, list)) and isinstance(
            points_y, (tuple, list)
        ), "Pose must be a list of 2 lists."
        assert len(points_x) == len(points_y) == 17, (
            "Pose x and y coordinates must have the same length of 17."
        )
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Pose coordinates must be floats or integers."

        if normalized_to is not None:
            assert all(0 <= coord <= 1 for coord in [*points_x, *points_y]), (
                "Normalized coordinates must be floats between 0 and 1."
            )
            width, height = validate_img_size(normalized_to)
            points_x = [coord * width for coord in points_x]
            points_y = [coord * height for coord in points_y]

        return Pose(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    def to_normalized(self, img_size: Sequence[int]) -> tuple[list[float], list[float]]:
        """
        Return the pose keypoints in normalized coordinates.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            tuple[list[float], list[float]]: The pose keypoints
                with normalized coordinates.
        """
        width, height = validate_img_size(img_size)
        assert all(x <= width and y <= height for x, y in zip(self.x, self.y)), (
            "Pose keypoints are out of image size."
        )
        return (
            [coord / width for coord in self.x],
            [coord / height for coord in self.y],
        )

    @staticmethod
    def from_dict(points: dict[str, Sequence[float]]) -> "Pose":
        assert isinstance(points, dict) and set(points) == {
            "x",
            "y",
        }, "Pose must be a dict with keys 'x' and 'y'."
        return Pose.from_list([points["x"], points["y"]])


class Pose3D(DataModel):
    """
    A data model for representing 3D pose keypoints.

    Attributes:
        x (list[int]): The x-coordinates of the keypoints.
        y (list[int]): The y-coordinates of the keypoints.
        visible (list[float]): The visibility of the keypoints.

    The keypoints are represented as lists of x, y, and visibility values,
    where each index corresponds to a specific body part.
    """

    x: list[int] = Field(default=[])
    y: list[int] = Field(default=[])
    visible: list[float] = Field(default=[])

    @staticmethod
    def from_list(
        points: Sequence[Sequence[float]],
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "Pose3D":
        """
        Create a Pose3D instance from a list of x, y coordinates and visibility values.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            points (Sequence[Sequence[float]]): The x and y coordinates
                of the keypoints. List of 3 lists: x, y coordinates
                and visibility values.
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            Pose3D: A Pose3D object.

        """
        assert isinstance(points, (tuple, list)), (
            "Pose3D must be a tuple or list of 3 lists."
        )
        assert len(points) == 3, (
            "Pose3D must be a list of 3 lists: x, y coordinates and visible."
        )
        points_x, points_y, points_v = points
        assert (
            isinstance(points_x, (tuple, list))
            and isinstance(points_y, (tuple, list))
            and isinstance(points_v, (tuple, list))
        ), "Pose3D must be a tuple or list of 3 lists."
        assert len(points_x) == len(points_y) == len(points_v) == 17, (
            "Pose3D x, y coordinates and visible must have the same length of 17."
        )
        assert all(
            isinstance(value, (int, float))
            for value in [*points_x, *points_y, *points_v]
        ), "Pose3D coordinates must be floats or integers."

        if normalized_to is not None:
            assert all(0 <= coord <= 1 for coord in [*points_x, *points_y]), (
                "Normalized coordinates must be floats between 0 and 1."
            )
            width, height = validate_img_size(normalized_to)
            points_x = [coord * width for coord in points_x]
            points_y = [coord * height for coord in points_y]

        return Pose3D(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
            visible=list(points_v),
        )

    def to_normalized(
        self,
        img_size: Sequence[int],
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Return the pose 3D keypoints in normalized coordinates.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            tuple[list[float], list[float], list[float]]: The pose keypoints
                with normalized coordinates and visibility values.
        """
        width, height = validate_img_size(img_size)
        assert all(x <= width and y <= height for x, y in zip(self.x, self.y)), (
            "Pose3D keypoints are out of image size."
        )
        return (
            [coord / width for coord in self.x],
            [coord / height for coord in self.y],
            self.visible,
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]]) -> "Pose3D":
        assert isinstance(points, dict) and set(points) == {
            "x",
            "y",
            "visible",
        }, "Pose3D must be a dict with keys 'x', 'y' and 'visible'."
        return Pose3D.from_list([points["x"], points["y"], points["visible"]])
