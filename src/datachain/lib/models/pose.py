from pydantic import Field

from datachain.lib.data_model import DataModel


class Pose(DataModel):
    """
    A data model for representing pose keypoints.

    Attributes:
        x (list[int]): The x-coordinates of the keypoints.
        y (list[int]): The y-coordinates of the keypoints.

    The keypoints are represented as lists of x and y coordinates, where each index
    corresponds to a specific body part.
    """

    x: list[int] = Field(default=None)
    y: list[int] = Field(default=None)

    @staticmethod
    def from_list(points: list[list[float]]) -> "Pose":
        assert len(points) == 2, "Pose coordinates must be a list of 2 lists."
        points_x, points_y = points
        assert (
            len(points_x) == len(points_y) == 17
        ), "Pose x and y coordinates must have the same length of 17."
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Pose coordinates must be integers or floats."
        return Pose(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]]) -> "Pose":
        assert set(points) == {"x", "y"}, "Pose coordinates must contain keys 'x' and 'y'."
        points_x, points_y = points["x"], points["y"]
        assert (
            len(points_x) == len(points_y) == 17
        ), "Pose x and y coordinates must have the same length of 17."
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Pose coordinates must be integers or floats."
        return Pose(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )


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

    x: list[int] = Field(default=None)
    y: list[int] = Field(default=None)
    visible: list[float] = Field(default=None)

    @staticmethod
    def from_list(points: list[list[float]]) -> "Pose3D":
        assert len(points) == 3, "Pose coordinates must be a list of 3 lists."
        points_x, points_y, points_v = points
        assert (
            len(points_x) == len(points_y) == len(points_v) == 17
        ), "Pose x, y, and visibility coordinates must have the same length of 17."
        assert all(
            isinstance(value, (int, float))
            for value in [*points_x, *points_y, *points_v]
        ), "Pose coordinates must be integers or floats."
        return Pose3D(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
            visible=points_v,
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]]) -> "Pose3D":
        assert set(points) == {"x", "y", "visible"}, "Pose coordinates must contain keys 'x', 'y', and 'visible'."
        points_x, points_y, points_v = points["x"], points["y"], points["visible"]
        assert (
            len(points_x) == len(points_y) == len(points_v) == 17
        ), "Pose x, y, and visibility coordinates must have the same length of 17."
        assert all(
            isinstance(value, (int, float))
            for value in [*points_x, *points_y, *points_v]
        ), "Pose coordinates must be integers or floats."
        return Pose3D(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
            visible=points_v,
        )
