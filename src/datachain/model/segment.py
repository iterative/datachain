from pydantic import Field

from datachain.lib.data_model import DataModel


class Segment(DataModel):
    """
    A data model for representing segment.

    Use `datachain.model.YoloSeg` or for YOLO-specific segments.
    This model is intended for general pose representations or other formats.

    Attributes:
        title (str): The title of the segment.
        x (list[float]): The x-coordinates of the segment.
        y (list[float]): The y-coordinates of the segment.

    The segment is represented as lists of x and y coordinates, where each index
    corresponds to a specific point.
    """

    title: str = Field(default="")
    x: list[float] = Field(default=[])
    y: list[float] = Field(default=[])
