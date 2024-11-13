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
from typing import TYPE_CHECKING, Optional

from PIL import Image
from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.model_store import ModelStore
from datachain.lib.models.bbox import BBox, OBBox

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO

    from datachain.lib.file import File


class YoloBBox(DataModel):
    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: Optional[BBox] = Field(default=None)

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
        box = None
        if "box" in summary[0]:
            box = BBox.from_dict(summary[0]["box"])
        return YoloBBox(
            cls=summary[0]["class"],
            name=summary[0]["name"],
            confidence=summary[0]["confidence"],
            box=box,
        )


class YoloBBoxes(DataModel):
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
        cls, name, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                cls.append(s["class"])
                name.append(s["name"])
                confidence.append(s["confidence"])
                box.append(BBox.from_dict(s.get("box", {})))
        return YoloBBoxes(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
        )


class YoloOBBox(DataModel):
    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: Optional[OBBox] = Field(default=None)

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
        box = None
        if "box" in summary[0]:
            box = OBBox.from_dict(summary[0]["box"])
        return YoloOBBox(
            cls=summary[0]["class"],
            name=summary[0]["name"],
            confidence=summary[0]["confidence"],
            box=box,
        )


class YoloOBBoxes(DataModel):
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
        cls, name, confidence, box = [], [], [], []
        for r in results:
            for s in r.summary():
                cls.append(s["class"])
                name.append(s["name"])
                confidence.append(s["confidence"])
                box.append(OBBox.from_dict(s.get("box", {})))
        return YoloOBBoxes(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
        )


ModelStore.register(YoloBBox)
ModelStore.register(YoloBBoxes)
ModelStore.register(YoloOBBox)
ModelStore.register(YoloOBBoxes)
