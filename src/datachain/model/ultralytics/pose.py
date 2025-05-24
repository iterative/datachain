from typing import TYPE_CHECKING

from pydantic import Field

from datachain.lib.data_model import DataModel
from datachain.model.bbox import BBox
from datachain.model.pose import Pose3D

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
        pose: The 3D pose keypoints.
    """

    cls: int = Field(default=-1)
    name: str = Field(default="")
    confidence: float = Field(default=0)
    box: BBox = Field(default=BBox())
    pose: Pose3D = Field(default=Pose3D())

    @staticmethod
    def from_result(result: "Results") -> "YoloPose":
        summary = result.summary()
        if not summary:
            return YoloPose(box=BBox(), pose=Pose3D())
        name = summary[0].get("name", "")
        box = (
            BBox.from_dict(summary[0]["box"], title=name)
            if summary[0].get("box")
            else BBox()
        )
        pose = (
            Pose3D.from_dict(summary[0]["keypoints"])
            if summary[0].get("keypoints")
            else Pose3D()
        )
        return YoloPose(
            cls=summary[0]["class"],
            name=name,
            confidence=summary[0]["confidence"],
            box=box,
            pose=pose,
        )


class YoloPoses(DataModel):
    """
    A data model for a list of YOLO pose keypoints.

    Attributes:
        cls: The classes of the poses.
        name: The names of the poses.
        confidence: The confidence scores of the poses.
        box: The bounding boxes of the poses.
        pose: The 3D pose keypoints of the poses.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    box: list[BBox] = Field(default=[])
    pose: list[Pose3D] = Field(default=[])

    @staticmethod
    def from_results(results: list["Results"]) -> "YoloPoses":
        cls, names, confidence, box, pose = [], [], [], [], []
        for r in results:
            for s in r.summary():
                name = s.get("name", "")
                cls.append(s["class"])
                names.append(name)
                confidence.append(s["confidence"])
                if s.get("box"):
                    box.append(BBox.from_dict(s.get("box"), title=name))
                if s.get("keypoints"):
                    pose.append(Pose3D.from_dict(s.get("keypoints")))
        return YoloPoses(
            cls=cls,
            name=names,
            confidence=confidence,
            box=box,
            pose=pose,
        )
