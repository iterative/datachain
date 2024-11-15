"""
This module contains the YOLO models.

YOLO stands for "You Only Look Once", a family of object detection models that
are designed to be fast and accurate. The models are trained to detect objects
in images by dividing the image into a grid and predicting the bounding boxes
and class probabilities for each grid cell.

More information about YOLO can be found here:
- https://pjreddie.com/darknet/yolo/
- https://docs.ultralytics.com/
"""

from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image
from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.models.bbox import BBox, OBBox

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO

    from datachain.lib.file import File


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
    box: BBox = Field(default=None)

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloBBox":
        results = yolo(Image.open(BytesIO(file.read())))
        if len(results) == 0:
            return YoloBBox()
        return YoloBBox.from_result(results[0])

    @staticmethod
    def from_result(result: "Results") -> "YoloBBox":
        summary = result.summary()
        if not summary:
            return YoloBBox()
        name = summary[0].get("name", "")
        box = (
            BBox.from_dict(summary[0]["box"], title=name)
            if "box" in summary[0]
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

    cls: list[int]
    name: list[str]
    confidence: list[float]
    box: list[BBox]

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloBBoxes":
        results = yolo(Image.open(BytesIO(file.read())))
        return YoloBBoxes.from_results(results)

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloBBoxes":
        cls, names, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                box.append(BBox.from_dict(s.get("box", {}), title=name))
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
    box: OBBox = Field(default=None)

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloOBBox":
        results = yolo(Image.open(BytesIO(file.read())))
        if len(results) == 0:
            return YoloOBBox()
        return YoloOBBox.from_result(results[0])

    @staticmethod
    def from_result(result: "Results") -> "YoloOBBox":
        summary = result.summary()
        if not summary:
            return YoloOBBox()
        name = summary[0].get("name", "")
        box = (
            OBBox.from_dict(summary[0]["box"], title=name)
            if "box" in summary[0]
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

    cls: list[int]
    name: list[str]
    confidence: list[float]
    box: list[OBBox]

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloOBBoxes":
        results = yolo(Image.open(BytesIO(file.read())))
        return YoloOBBoxes.from_results(results)

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloOBBoxes":
        cls, names, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                box.append(OBBox.from_dict(s.get("box", {}), title=name))
        return YoloOBBoxes(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
        )
