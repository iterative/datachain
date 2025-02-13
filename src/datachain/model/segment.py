from collections.abc import Sequence
from typing import Optional

from pydantic import Field

from datachain.lib.data_model import DataModel

from .utils import validate_img_size


class Segment(DataModel):
    """
    A data model for representing segment.

    Attributes:
        title (str): The title of the segment.
        x (list[int]): The x-coordinates of the segment.
        y (list[int]): The y-coordinates of the segment.

    The segment is represented as lists of x and y coordinates, where each index
    corresponds to a specific point.
    """

    title: str = Field(default="")
    x: list[int] = Field(default=[])
    y: list[int] = Field(default=[])

    @staticmethod
    def from_list(
        points: Sequence[Sequence[float]],
        title: str = "",
        normalized_to: Optional[Sequence[int]] = None,
    ) -> "Segment":
        """
        Create a Segment object from a list of x and y coordinates.

        If the input coordinates are normalized (i.e., floats between 0 and 1),
        they will be converted to absolute pixel values based on the provided
        image size. The image size should be given as a tuple (width, height)
        via the `normalized_to` argument.

        Args:
            points (Sequence[Sequence[float]]): The x and y coordinates
                of the keypoints. List of 2 lists: x and y coordinates.
            title (str, optional): The title or label for the segment. Defaults to "".
            normalized_to (Sequence[int], optional): The reference image size
                (width, height) for denormalizing the bounding box. If None (default),
                the coordinates are assumed to be absolute pixel values.

        Returns:
            Segment: A Segment object.
        """
        assert isinstance(points, (tuple, list)), "Segment must be a list of 2 lists."
        assert len(points) == 2, (
            "Segment must be a list of 2 lists: x and y coordinates."
        )
        points_x, points_y = points
        assert isinstance(points_x, (tuple, list)) and isinstance(
            points_y, (tuple, list)
        ), "Segment must be a list of 2 lists."
        assert len(points_x) == len(points_y), (
            "Segment x and y coordinates must have the same length."
        )
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Segment coordinates must be floats or integers."

        if normalized_to is not None:
            assert all(0 <= coord <= 1 for coord in [*points_x, *points_y]), (
                "Normalized coordinates must be floats between 0 and 1."
            )
            width, height = validate_img_size(normalized_to)
            points_x = [coord * width for coord in points_x]
            points_y = [coord * height for coord in points_y]

        return Segment(
            title=title,
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    def to_normalized(self, img_size: Sequence[int]) -> tuple[list[float], list[float]]:
        """
        Return the segment in normalized coordinates.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            tuple[list[float], list[float]]: The segment with normalized coordinates.
        """
        width, height = validate_img_size(img_size)
        assert all(x <= width and y <= height for x, y in zip(self.x, self.y)), (
            "Segment keypoints are out of image size."
        )
        return (
            [coord / width for coord in self.x],
            [coord / height for coord in self.y],
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]], title: str = "") -> "Segment":
        assert isinstance(points, dict) and set(points) == {
            "x",
            "y",
        }, "Segment must be a dict with keys 'x' and 'y'."
        return Segment.from_list([points["x"], points["y"]], title=title)
