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
    def from_list(points: list[list[float]], title: str = "") -> "Segment":
        assert (
            len(points) == 2
        ), "Segment must be a list of 2 lists: x and y coordinates."
        points_x, points_y = points
        assert len(points_x) == len(
            points_y
        ), "Segment x and y coordinates must have the same length."
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Segment coordinates must be floats or integers."
        return Segment(
            title=title,
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]], title: str = "") -> "Segment":
        assert isinstance(points, dict) and set(points) == {
            "x",
            "y",
        }, "Segment must be a dict with keys 'x' and 'y'."
        return Segment.from_list([points["x"], points["y"]], title=title)
