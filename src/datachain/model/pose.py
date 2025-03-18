from collections.abc import Sequence

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
    def from_list(points: Sequence[Sequence[float]]) -> "Pose":
        if not isinstance(points, (list, tuple)):
            raise TypeError("Pose must be a list of coordinates.")
        if len(points) != 2:
            raise ValueError("Pose must be a list of 2 lists: x and y coordinates.")
        points_x, points_y = points
        if not isinstance(points_x, (list, tuple)) or not isinstance(
            points_y, (list, tuple)
        ):
            raise TypeError("Pose x and y coordinates must be lists.")
        if len(points_x) != len(points_y) != 17:
            raise ValueError(
                "Pose x and y coordinates must have the same length of 17."
            )
        if not all(isinstance(value, (int, float)) for value in [*points_x, *points_y]):
            raise ValueError("Pose coordinates must be floats or integers.")
        return Pose(x=list(map(round, points_x)), y=list(map(round, points_y)))

    @staticmethod
    def from_dict(points: dict[str, Sequence[float]]) -> "Pose":
        keys = ("x", "y")
        if not isinstance(points, dict) or set(points) != set(keys):
            raise ValueError("Pose must be a dictionary with coordinates.")
        return Pose.from_list([points[k] for k in keys])


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
    def from_list(points: Sequence[Sequence[float]]) -> "Pose3D":
        if not isinstance(points, (list, tuple)):
            raise TypeError("Pose3D must be a list of coordinates.")
        if len(points) != 3:
            raise ValueError(
                "Pose3D must be a list of 3 lists: x, y coordinates and visible."
            )
        points_x, points_y, points_v = points
        if (
            not isinstance(points_x, (list, tuple))
            or not isinstance(points_y, (list, tuple))
            or not isinstance(points_v, (list, tuple))
        ):
            raise TypeError("Pose3D x, y and visible must be lists.")
        if len(points_x) != len(points_y) != len(points_v) != 17:
            raise ValueError("Pose3D x, y and visible must have the same length of 17.")
        if not all(
            isinstance(value, (int, float))
            for value in [*points_x, *points_y, *points_v]
        ):
            raise ValueError("Pose3D coordinates must be floats or integers.")
        return Pose3D(
            x=list(map(round, points_x)),
            y=list(map(round, points_y)),
            visible=list(points_v),
        )

    @staticmethod
    def from_dict(points: dict[str, Sequence[float]]) -> "Pose3D":
        keys = ("x", "y", "visible")
        if not isinstance(points, dict) or set(points) != set(keys):
            raise ValueError("Pose3D must be a dictionary with coordinates.")
        return Pose3D.from_list([points[k] for k in keys])
