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

from typing import TYPE_CHECKING

from pydantic import Field

from datachain.lib.data_model import DataModel
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
    """
    A data model for YOLO pose keypoints.

    Attributes:
        cls: The class of the pose.
        name: The name of the pose.
        confidence: The confidence score of the pose.
        box: The bounding box of the pose.
        keypoints: The 3D pose keypoints.
    """

    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: BBox = Field(default=None)
    keypoints: Pose3D = Field(default=None)

    @staticmethod
    def from_result(result: "Results") -> "YoloPose":
        summary = result.summary()
        if not summary:
            return YoloPose()
        name = summary[0].get("name", "")
        box = (
            BBox.from_dict(summary[0]["box"], title=name)
            if "box" in summary[0]
            else BBox()
        )
        keypoints = (
            Pose3D.from_dict(summary[0]["keypoints"])
            if "keypoints" in summary[0]
            else Pose3D()
        )
        return YoloPose(
            cls=summary[0]["class"],
            name=name,
            confidence=summary[0]["confidence"],
            box=box,
            keypoints=keypoints,
        )


class YoloPoses(DataModel):
    """
    A data model for a list of YOLO pose keypoints.

    Attributes:
        cls: The classes of the poses.
        name: The names of the poses.
        confidence: The confidence scores of the poses.
        box: The bounding boxes of the poses.
        keypoints: The 3D pose keypoints of the poses.
    """

    cls: list[int]
    name: list[str]
    confidence: list[float]
    box: list[BBox]
    keypoints: list[Pose3D]

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloPoses":
        cls, names, confidence, box, keypoints = [], [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                box.append(BBox.from_dict(s.get("box", {}), title=name))
                keypoints.append(Pose3D.from_dict(s.get("keypoints", {})))
        return YoloPoses(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
            keypoints=keypoints,
        )
