from pydantic import Field

from datachain.lib.data_model import DataModel


class Segments(DataModel):
    """
    A data model for representing segments.

    Attributes:
        title (str): The title of the segments.
        x (list[int]): The x-coordinates of the segments.
        y (list[int]): The y-coordinates of the segments.

    The segments are represented as lists of x and y coordinates, where each index
    corresponds to a specific segment.
    """

    title: str = Field(default="")
    x: list[int] = Field(default=None)
    y: list[int] = Field(default=None)

    @staticmethod
    def from_list(points: list[list[float]], title: str = "") -> "Segments":
        assert len(points) == 2, "Segments coordinates must be a list of 2 lists."
        points_x, points_y = points
        assert len(points_x) == len(
            points_y
        ), "Segments x and y coordinates must have the same length."
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Segments coordinates must be integers or floats."
        return Segments(
            title=title,
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]], title: str = "") -> "Segments":
        assert set(points) == {
            "x",
            "y",
        }, "Segments coordinates must contain keys 'x' and 'y'."
        points_x, points_y = points["x"], points["y"]
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Segments coordinates must be integers or floats."
        return Segments(
            title=title,
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )
