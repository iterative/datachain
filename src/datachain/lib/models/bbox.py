from typing import Optional

from pydantic import Field

from datachain.lib.data_model import DataModel


class BBox(DataModel):
    """
    A data model for representing bounding boxes.

    Attributes:
        title (str): The title of the bounding box.
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (float): The y-coordinate of the bottom-right corner of the bounding box.

    The bounding box is defined by two points:
        - (x1, y1): The top-left corner of the box.
        - (x2, y2): The bottom-right corner of the box.
    """

    title: str = Field(default="")
    x1: float = Field(default=0)
    y1: float = Field(default=0)
    x2: float = Field(default=0)
    y2: float = Field(default=0)

    @staticmethod
    def from_xywh(bbox: list[float], title: Optional[str] = None) -> "BBox":
        """
        Converts a bounding box in (x, y, width, height) format
        to a BBox data model instance.

        Args:
            bbox (list[float]): A bounding box, represented as a list
                                of four floats [x, y, width, height].

        Returns:
            BBox2D: An instance of the BBox data model.
        """
        assert len(bbox) == 4, f"Bounding box must have 4 elements, got f{len(bbox)}"
        x, y, w, h = bbox
        return BBox(title=title or "", x1=x, y1=y, x2=x + w, y2=y + h)
