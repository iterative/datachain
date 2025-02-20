from collections.abc import Sequence
from warnings import warn

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.model.utils import BBoxType, convert_bbox


class BBox(DataModel):
    """
    A data model for representing bounding box.

    Use `datachain.model.Yolo` or for YOLO-specific bounding boxes.
    This model is intended for general bounding box representations or other formats.

    Attributes:
        title (str): The title of the bounding box.
        coords (list[float]): The coordinates of the bounding box.
    """

    title: str = Field(default="")
    coords: list[float] = Field(default=[])

    @staticmethod
    def from_list(coords: Sequence[float], title: str = "") -> "BBox":
        """
        Create a bounding box from a list of coordinates.

        Args:
            coords (list[float]): The bounding box coordinates.
            title (str): The title of the bounding box.

        Returns:
            BBox: The bounding box instance.
        """
        warn(
            "This method is deprecated. Use `BBox(title, coords)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("Bounding box must be a list of 4 coordinates.")
        if not all(isinstance(value, (int, float)) for value in coords):
            raise ValueError("Bounding box coordinates must be floats or integers.")
        return BBox(title=title, coords=list(map(float, coords)))

    def convert(
        self,
        img_size: Sequence[int],
        source: BBoxType,
        target: BBoxType,
    ) -> list[float]:
        """
        Convert the bounding box coordinates between different formats.

        Supported formats: "albumentations", "coco", "voc", "yolo".

        Args:
            img_size (Sequence[int]): The reference image size (width, height).
            source (str): The source bounding box format.
            target (str): The target bounding box format.

        Returns:
            list[float]: The bounding box coordinates in the target format.
        """
        return convert_bbox(self.coords, img_size, source, target)


class OBBox(DataModel):
    """
    A data model for representing oriented bounding boxes.

    Use `datachain.model.YoloObb` for YOLO-specific oriented bounding boxes.
    This model is intended for general oriented bounding box representations
    or other formats.

    Attributes:
        title (str): The title of the oriented bounding box.
        coords (list[float]): The coordinates of the oriented bounding box.
    """

    title: str = Field(default="")
    coords: list[float] = Field(default=[])
