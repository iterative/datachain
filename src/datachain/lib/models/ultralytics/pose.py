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

from typing import TYPE_CHECKING, Optional

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.lib.model_store import ModelStore
from datachain.lib.models.bbox import BBox
from datachain.lib.models.pose import Pose3D

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class YoloPoseBodyPart:
    """An enumeration of body parts for YOLO pose keypoints."""

    nose = 0
    left_eye = 1
    right_eye = 2
    left_ear = 3
    right_ear = 4
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    left_wrist = 9
    right_wrist = 10
    left_hip = 11
    right_hip = 12
    left_knee = 13
    right_knee = 14
    left_ankle = 15
    right_ankle = 16


class YoloPose(DataModel):
    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: Optional[BBox] = Field(default=None)
    keypoints: Pose3D = Field(default=None)

    @staticmethod
    def from_result(result: "Results") -> "YoloPose":
        summary = result.summary()
        if not summary:
            return YoloPose()
        box, keypoints = None, Pose3D()
        if "box" in summary[0]:
            box = BBox.from_dict(summary[0]["box"])
        if "keypoints" in summary[0]:
            keypoints = Pose3D.from_dict(summary[0]["keypoints"])
        return YoloPose(
            cls=summary[0]["class"],
            name=summary[0]["name"],
            confidence=summary[0]["confidence"],
            box=box,
            keypoints=keypoints,
        )


class YoloPoses(DataModel):
    cls: list[int]
    name: list[str]
    confidence: list[float]
    box: list[BBox]
    keypoints: list[Pose3D]

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloPoses":
        cls, name, confidence, box, keypoints = [], [], [], [], []
        for r in results:
            for s in r.summary():
                cls.append(s["class"])
                name.append(s["name"])
                confidence.append(s["confidence"])
                box.append(BBox.from_dict(s.get("box", {})))
                keypoints.append(Pose3D.from_dict(s.get("keypoints", {})))
        return YoloPoses(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
            keypoints=keypoints,
        )


ModelStore.register(YoloPose)
ModelStore.register(YoloPoses)
