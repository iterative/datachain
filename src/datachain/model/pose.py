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

    x: list[int] = Field(default=[])
    y: list[int] = Field(default=[])

    @staticmethod
    def from_list(points: list[list[float]]) -> "Pose":
        assert len(points) == 2, "Pose must be a list of 2 lists: x and y coordinates."
        points_x, points_y = points
        assert (
            len(points_x) == len(points_y) == 17
        ), "Pose x and y coordinates must have the same length of 17."
        assert all(
            isinstance(value, (int, float)) for value in [*points_x, *points_y]
        ), "Pose coordinates must be floats or integers."
        return Pose(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]]) -> "Pose":
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
    def from_list(points: list[list[float]]) -> "Pose3D":
        assert (
            len(points) == 3
        ), "Pose3D must be a list of 3 lists: x, y coordinates and visible."
        points_x, points_y, points_v = points
        assert (
            len(points_x) == len(points_y) == len(points_v) == 17
        ), "Pose3D x, y coordinates and visible must have the same length of 17."
        assert all(
            isinstance(value, (int, float))
            for value in [*points_x, *points_y, *points_v]
        ), "Pose3D coordinates must be floats or integers."
        return Pose3D(
            x=[round(coord) for coord in points_x],
            y=[round(coord) for coord in points_y],
            visible=points_v,
        )

    @staticmethod
    def from_dict(points: dict[str, list[float]]) -> "Pose3D":
        assert isinstance(points, dict) and set(points) == {
            "x",
            "y",
            "visible",
        }, "Pose3D must be a dict with keys 'x', 'y' and 'visible'."
        return Pose3D.from_list([points["x"], points["y"], points["visible"]])
