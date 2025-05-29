from typing import TYPE_CHECKING

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.model.bbox import BBox, OBBox

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class YoloBBox(DataModel):
    """
    A class representing a bounding box detected by a YOLO model.

    Attributes:
        cls: The class of the detected object.
        name: The name of the detected object.
        confidence: The confidence score of the detection.
        box: The bounding box of the detected object
    """

    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: BBox = Field(default=BBox())

    @staticmethod
    def from_result(result: "Results") -> "YoloBBox":
        summary = result.summary()
        if not summary:
            return YoloBBox(box=BBox())
        name = summary[0].get("name", "")
        box = (
            BBox.from_dict(summary[0]["box"], title=name)
            if summary[0].get("box")
            else BBox()
        )
        return YoloBBox(
            cls=summary[0]["class"],
            name=name,
            confidence=summary[0]["confidence"],
            box=box,
        )


class YoloBBoxes(DataModel):
    """
    A class representing a list of bounding boxes detected by a YOLO model.

    Attributes:
        cls: A list of classes of the detected objects.
        name: A list of names of the detected objects.
        confidence: A list of confidence scores of the detections.
        box: A list of bounding boxes of the detected objects
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    box: list[BBox] = Field(default=[])

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloBBoxes":
        cls, names, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                if s.get("box"):
                    box.append(BBox.from_dict(s.get("box"), title=name))
        return YoloBBoxes(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
        )


class YoloOBBox(DataModel):
    """
    A class representing an oriented bounding box detected by a YOLO model.

    Attributes:
        cls: The class of the detected object.
        name: The name of the detected object.
        confidence: The confidence score of the detection.
        box: The oriented bounding box of the detected object.
    """

    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: OBBox

    @staticmethod
    def from_result(result: "Results") -> "YoloOBBox":
        summary = result.summary()
        if not summary:
            return YoloOBBox(box=OBBox())
        name = summary[0].get("name", "")
        box = (
            OBBox.from_dict(summary[0]["box"], title=name)
            if summary[0].get("box")
            else OBBox()
        )
        return YoloOBBox(
            cls=summary[0]["class"],
            name=name,
            confidence=summary[0]["confidence"],
            box=box,
        )


class YoloOBBoxes(DataModel):
    """
    A class representing a list of oriented bounding boxes detected by a YOLO model.

    Attributes:
        cls: A list of classes of the detected objects.
        name: A list of names of the detected objects.
        confidence: A list of confidence scores of the detections.
        box: A list of oriented bounding boxes of the detected objects.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    box: list[OBBox] = Field(default=[])

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloOBBoxes":
        cls, names, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                if s.get("box"):
                    box.append(OBBox.from_dict(s.get("box"), title=name))
        return YoloOBBoxes(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
        )
