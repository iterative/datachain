from typing import TYPE_CHECKING

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.model.bbox import BBox
from datachain.model.segment import Segment

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class YoloSegment(DataModel):
    """
    A data model for a single YOLO segment.

    Attributes:
        cls (int): The class of the segment.
        name (str): The name of the segment.
        confidence (float): The confidence of the segment.
        box (BBox): The bounding box of the segment.
        segment (Segments): The segments of the segment.
    """

    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: BBox = Field(default=BBox())
    segment: Segment = Field(default=Segment())

    @staticmethod
    def from_result(result: "Results") -> "YoloSegment":
        summary = result.summary()
        if not summary:
            return YoloSegment(box=BBox(), segment=Segment())
        name = summary[0].get("name", "")
        box = (
            BBox.from_dict(summary[0]["box"], title=name)
            if summary[0].get("box")
            else BBox()
        )
        segment = (
            Segment.from_dict(summary[0]["segments"], title=name)
            if summary[0].get("segments")
            else Segment()
        )
        return YoloSegment(
            cls=summary[0]["class"],
            name=summary[0]["name"],
            confidence=summary[0]["confidence"],
            box=box,
            segment=segment,
        )


class YoloSegments(DataModel):
    """
    A data model for a list of YOLO segments.

    Attributes:
        cls (list[int]): The classes of the segments.
        name (list[str]): The names of the segments.
        confidence (list[float]): The confidences of the segments.
        box (list[BBox]): The bounding boxes of the segments.
        segment (list[Segments]): The segments of the segments.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    box: list[BBox] = Field(default=[])
    segment: list[Segment] = Field(default=[])

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloSegments":
        cls, names, confidence, box, segment = [], [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                if s.get("box"):
                    box.append(BBox.from_dict(s.get("box"), title=name))
                if s.get("segments"):
                    segment.append(Segment.from_dict(s.get("segments"), title=name))
        return YoloSegments(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
            segment=segment,
        )
