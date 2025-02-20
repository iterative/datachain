from pydantic import Field

from datachain.lib.data_model import DataModel


class Pose(DataModel):
    """
    A data model for representing pose keypoints.

    Use `datachain.model.YoloPose` or for YOLO-specific poses.
    This model is intended for general pose representations or other formats.

    Attributes:
        x (list[int]): The x-coordinates of the keypoints.
        y (list[int]): The y-coordinates of the keypoints.

    The keypoints are represented as lists of x and y coordinates, where each index
    corresponds to a specific body part.
    """

    x: list[float] = Field(default=[])
    y: list[float] = Field(default=[])


class Pose3D(DataModel):
    """
    A data model for representing 3D pose keypoints.

    Use `datachain.model.YoloPose` or for YOLO-specific poses.
    This model is intended for general pose representations or other formats.

    Attributes:
        x (list[float]): The x-coordinates of the keypoints.
        y (list[float]): The y-coordinates of the keypoints.
        visible (list[float]): The visibility of the keypoints.

    The keypoints are represented as lists of x and y coordinates and visibility,
    where each index corresponds to a specific body part.
    """

    x: list[float] = Field(default=[])
    y: list[float] = Field(default=[])
    visible: list[float] = Field(default=[])
