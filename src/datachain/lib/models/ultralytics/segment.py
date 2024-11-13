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
from datachain.lib.models.bbox import BBox
from datachain.lib.models.segment import Segments

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.models import YOLO

    from datachain.lib.file import File


class YoloSegment(DataModel):
    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: Optional[BBox] = Field(default=None)
    segments: Segments = Field(default=None)

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloSegment":
        results = yolo(Image.open(BytesIO(file.read())))
        if len(results) == 0:
            return YoloSegment()
        return YoloSegment.from_result(results[0])

    @staticmethod
    def from_result(result: "Results") -> "YoloSegment":
        summary = result.summary()
        if not summary:
            return YoloSegment()
        box, segments = None, Segments()
        if "box" in summary[0]:
            box = BBox.from_dict(summary[0]["box"])
        if "segments" in summary[0]:
            segments = Segments.from_dict(summary[0]["segments"])
        return YoloSegment(
            cls=summary[0]["class"],
            name=summary[0]["name"],
            confidence=summary[0]["confidence"],
            box=box,
            segments=segments,
        )


class YoloSegments(DataModel):
    cls: list[int]
    name: list[str]
    confidence: list[float]
    box: list[BBox]
    segments: list[Segments]

    @staticmethod
    def from_file(yolo: "YOLO", file: "File") -> "YoloSegments":
        results = yolo(Image.open(BytesIO(file.read())))
        return YoloSegments.from_results(results)

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloSegments":
        cls, name, confidence, box, segments = [], [], [], [], []
        for r in results:
            for s in r.summary():
                cls.append(s["class"])
                name.append(s["name"])
                confidence.append(s["confidence"])
                box.append(BBox.from_dict(s.get("box", {})))
                segments.append(Segments.from_dict(s.get("segments", {})))
        return YoloSegments(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
            segments=segments,
        )


ModelStore.register(YoloSegment)
ModelStore.register(YoloSegments)
