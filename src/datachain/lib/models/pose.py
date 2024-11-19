from pydantic import Field

from datachain.lib.data_model import DataModel


class Pose(DataModel):
    """
    A data model for representing pose keypoints.

    Attributes:
        x (list[float]): The x-coordinates of the keypoints.
        y (list[float]): The y-coordinates of the keypoints.

    The keypoints are represented as lists of x and y coordinates, where each index
    corresponds to a specific body part.
    """

    x: list[float] = Field(default=None)
    y: list[float] = Field(default=None)


class Pose3D(DataModel):
    """
    A data model for representing 3D pose keypoints.

    Attributes:
        x (list[float]): The x-coordinates of the keypoints.
        y (list[float]): The y-coordinates of the keypoints.
        visible (list[float]): The visibility of the keypoints.

    The keypoints are represented as lists of x, y, and visibility values,
    where each index corresponds to a specific body part.
    """

    x: list[float] = Field(default=None)
    y: list[float] = Field(default=None)
    visible: list[float] = Field(default=None)
