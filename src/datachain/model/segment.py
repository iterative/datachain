from collections.abc import Sequence

from pydantic import Field

from datachain.lib.data_model import DataModel


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
    def from_list(points: Sequence[Sequence[float]], title: str = "") -> "Segment":
        if not isinstance(points, (list, tuple)):
            raise TypeError("Segment must be a list of coordinates.")
        if len(points) != 2:
            raise ValueError("Segment must be a list of 2 lists: x and y coordinates.")
        points_x, points_y = points
        if not isinstance(points_x, (list, tuple)) or not isinstance(
            points_y, (list, tuple)
        ):
            raise TypeError("Segment x and y coordinates must be lists.")
        if len(points_x) != len(points_y):
            raise ValueError("Segment x and y coordinates must have the same length.")
        if not all(isinstance(value, (int, float)) for value in [*points_x, *points_y]):
            raise ValueError("Segment coordinates must be floats or integers.")
        return Segment(
            title=title,
            x=list(map(round, points_x)),
            y=list(map(round, points_y)),
        )

    @staticmethod
    def from_dict(points: dict[str, Sequence[float]], title: str = "") -> "Segment":
        keys = ("x", "y")
        if not isinstance(points, dict) or set(points) != set(keys):
            raise ValueError("Segment must be a dictionary with coordinates.")
        return Segment.from_list([points[k] for k in keys], title=title)
