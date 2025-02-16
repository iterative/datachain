"""
This module contains the YOLO models.

YOLO stands for "You Only Look Once", a family of object detection models that
are designed to be fast and accurate. The models are trained to detect objects
in images by dividing the image into a grid and predicting the bounding boxes
and class probabilities for each grid cell.

More information about YOLO can be found here:
- https://docs.ultralytics.com/
- https://docs.ultralytics.com/models/yolo11/
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import Field

from datachain.lib.data_model import DataModel

from .utils import convert_bbox

if TYPE_CHECKING:
    from ultralytics.engine.results import Results


class YoloBox(DataModel):
    """
    A class representing objects bounding boxes detected by a YOLO model.

    Object detection is a task that involves identifying the location and class
    of objects in an image or video stream.

    See https://docs.ultralytics.com/tasks/detect/ for more information.

    Attributes:
        cls (list[int]): A list of classes of the detected objects, default `[]`.
        name (list[str]): A list of names of the detected objects, default `[]`.
        confidence (list[float]): A list of confidence scores of the detections,
            default `[]`.
        box (list[list[float]]): A list of bounding boxes of the detected objects,
            stored as pixels coordinates [x_min, y_min, x_max, y_max]
            (PASCAL VOC format), default `[]`.
        orig_shape (list[int]): The original size of the image (height, width),
            default `[]`.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    box: list[list[float]] = Field(default=[])
    orig_shape: list[int] = Field(default=[])

    @staticmethod
    def from_yolo_results(results: list["Results"]) -> "YoloBox":
        """
        Create a YOLO bounding boxes from the YOLO results.

        Example:
            ```python
            from ultralytics import YOLO
            from datachain.model.bbox import YoloBox

            model = YOLO("yolo11n.pt")
            results = model("image.jpg", verbose=False)
            boxes = YoloBox.from_yolo_results(results)
            ```

        Args:
            results: YOLO results from the model.

        Returns:
            YoloBox: A YOLO bounding boxes data model.
        """
        if not (summary := results[0].summary(normalize=False)):
            return YoloBox()

        cls, name, confidence, box = [], [], [], []
        for res in summary:
            cls.append(res.get("class", -1))
            name.append(res.get("name", ""))
            confidence.append(res.get("confidence", -1))
            box.append(_get_box_from_yolo_result(res))

        return YoloBox(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
            orig_shape=list(results[0].orig_shape),
        )

    @property
    def img_size(self) -> tuple[int, int]:
        """Get the image size (width, height) from the original shape."""
        return (
            (self.orig_shape[1], self.orig_shape[0])
            if len(self.orig_shape) == 2
            else (0, 0)
        )

    def to_albumentations(self) -> Iterator[list[float]]:
        """
        Convert the bounding box to Albumentations format with normalized coordinates.

        Albumentations format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
            - (x_min, y_min) are the normalized coordinates of the top-left corner.
            - (x_max, y_max) are the normalized coordinates of the bottom-right corner.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            Iterator[list[float]]: An iterator of bounding box coordinates
                in Albumentations format with normalized coordinates.
        """
        return (
            convert_bbox(b, self.img_size, source="voc", target="albumentations")
            for b in self.box
        )

    def to_coco(self) -> Iterator[list[int]]:
        """
        Convert the bounding box to COCO format.

        COCO format represents bounding boxes as [x_min, y_min, width, height], where:
            - (x_min, y_min) are the pixel coordinates of the top-left corner.
            - width and height define the size of the bounding box in pixels.

        Returns:
            Iterator[list[int]]: An iterator of bounding box coordinates in COCO format.
        """
        return (
            list(
                map(round, convert_bbox(b, self.img_size, source="voc", target="coco"))
            )
            for b in self.box
        )

    def to_voc(self) -> Iterator[list[int]]:
        """
        Convert the bounding box to PASCAL VOC format.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
            - (x_min, y_min) are the pixel coordinates of the top-left corner.
            - (x_max, y_max) are the pixel coordinates of the bottom-right corner.

        Returns:
            Iterator[list[int]]: An iterator of bounding box coordinates
                in PASCAL VOC format.
        """
        return (
            list(map(round, convert_bbox(b, self.img_size, source="voc", target="voc")))
            for b in self.box
        )

    def to_yolo(self) -> Iterator[list[float]]:
        """
        Convert the bounding box to YOLO format with normalized coordinates.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
            - (x_center, y_center) are the normalized coordinates of the box center.
            - width and height normalized values define the size of the bounding box.

        Normalized coordinates are floats between 0 and 1, representing the
        relative position of the pixels in the image.

        Returns:
            Iterator[list[float]]: An iterator of bounding box coordinates
                in YOLO format with normalized coordinates.
        """
        return (
            convert_bbox(b, self.img_size, source="voc", target="yolo")
            for b in self.box
        )


class YoloObb(DataModel):
    """
    A class representing objects oriented bounding boxes detected by a YOLO model.

    Oriented object detection goes a step further than object detection and introduce
    an extra angle to locate objects more accurate in an image.

    See https://docs.ultralytics.com/tasks/obb/ for more information.

    Attributes:
        cls (list[int]): A list of classes of the detected objects, default `[]`.
        name (list[str]): A list of names of the detected objects, default `[]`.
        confidence (list[float]): A list of confidence scores of the detections,
            default `[]`.
        obox (list[list[float]]): A list of oriented bounding boxes of the detected
            objects, stored as four corners pixels coordinates
            [x1, y1, x2, y2, x3, y3, x4, y4], default `[]`.
        orig_shape (list[int]): The original size of the image (height, width),
            default `[]`.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])
    obox: list[list[float]] = Field(default=[])
    orig_shape: list[int] = Field(default=[])

    @staticmethod
    def from_yolo_results(results: list["Results"]) -> "YoloObb":
        """
        Create a YOLO oriented bounding boxes from the YOLO results.

        Example:
            ```python
            from ultralytics import YOLO
            from datachain.model.bbox import YoloObb

            model = YOLO("yolo11n-obb.pt")
            results = model("image.jpg", verbose=False)
            boxes = YoloObb.from_yolo_results(results)
            ```

        Args:
            results: YOLO results from the model.

        Returns:
            YoloObb: A YOLO oriented bounding boxes data model.
        """
        if not (summary := results[0].summary(normalize=False)):
            return YoloObb()

        cls, name, confidence, obox = [], [], [], []
        for res in summary:
            cls.append(res.get("class", -1))
            name.append(res.get("name", ""))
            confidence.append(res.get("confidence", -1))
            obox.append(_get_obox_from_yolo_result(res))

        return YoloObb(
            cls=cls,
            name=name,
            confidence=confidence,
            obox=obox,
            orig_shape=list(results[0].orig_shape),
        )

    @property
    def img_size(self) -> tuple[int, int]:
        """Get the image size (width, height) from the original shape."""
        return (
            (self.orig_shape[1], self.orig_shape[0])
            if len(self.orig_shape) == 2
            else (0, 0)
        )


class YoloSeg(YoloBox):
    """
    A class representing objects segmentation detected by a YOLO model.

    This class extends the `YoloBox` class to include the segments
    of the detected objects.

    Instance segmentation goes a step further than object detection and involves
    identifying individual objects in an image and segmenting them
    from the rest of the image.

    See https://docs.ultralytics.com/tasks/segment/ for more information.

    Attributes:
        cls (list[int]): A list of classes of the detected objects, default `[]`.
        name (list[str]): A list of names of the detected objects, default `[]`.
        confidence (list[float]): A list of confidence scores of the detections,
            default `[]`.
        box (list[list[float]]): A list of bounding boxes of the detected objects,
            stored as pixels coordinates [x_min, y_min, x_max, y_max]
            (PASCAL VOC format), default `[]`.
        segments (list[list[list[float]]]): A list of segments of the detected objects,
            stored as a list of x and y coordinates, default `[]`.
        orig_shape (list[int]): The original size of the image (height, width),
            default `[]`.
    """

    segments: list[list[list[float]]] = Field(default=[])

    @staticmethod
    def from_yolo_results(results: list["Results"]) -> "YoloSeg":
        """
        Create a YOLO oriented bounding boxes from the YOLO results.

        Example:
            ```python
            from ultralytics import YOLO
            from datachain.model.bbox import YoloSeg

            model = YOLO("yolo11n-seg.pt")
            results = model("image.jpg", verbose=False)
            segments = YoloSeg.from_yolo_results(results)
            ```

        Args:
            results: YOLO results from the model.

        Returns:
            YoloSeg: A YOLO segmentation data model.
        """
        if not (summary := results[0].summary(normalize=False)):
            return YoloSeg()

        cls, name, confidence, box, segments = [], [], [], [], []
        for res in summary:
            cls.append(res.get("class", -1))
            name.append(res.get("name", ""))
            confidence.append(res.get("confidence", -1))
            box.append(_get_box_from_yolo_result(res))
            segments.append(_get_segments_from_yolo_result(res))

        return YoloSeg(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
            segments=segments,
            orig_shape=list(results[0].orig_shape),
        )


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


class YoloPose(YoloBox):
    """
    A class representing human pose keypoints detected by a YOLO model.

    This class extends the `YoloBox` class to include the segments
    of the detected objects.

    Pose estimation is a task that involves identifying the location of specific points
    in an image, usually referred to as keypoints.

    See https://docs.ultralytics.com/tasks/pose/ for more information.

    Attributes:
        cls (list[int]): A list of classes of the detected objects, default `[]`.
        name (list[str]): A list of names of the detected objects, default `[]`.
        confidence (list[float]): A list of confidence scores of the detections,
            default `[]`.
        box (list[list[float]]): A list of bounding boxes of the detected objects,
            stored as pixels coordinates [x_min, y_min, x_max, y_max]
            (PASCAL VOC format), default `[]`.
        keypoints (list[list[list[float]]]): A list of human pose keypoints
            of the detected objects, stored as a list of x and y coordinates
            and visibility score, default `[]`.
        orig_shape (list[int]): The original size of the image (height, width),
            default `[]`.

    Note:
        There are 17 keypoints in total, each represented by a pair of x and y
        coordinates and a visibility score. The keypoints can be accessed by name
        using the `datachain.model.YoloPoseBodyPart` enumeration.
    """

    keypoints: list[list[list[float]]] = Field(default=[])

    @staticmethod
    def from_yolo_results(results: list["Results"]) -> "YoloPose":
        """
        Create a YOLO pose keypoints from the YOLO results.

        Example:
            ```python
            from ultralytics import YOLO
            from datachain.model.bbox import YoloPose

            model = YOLO("yolo11n-pose.pt")
            results = model("image.jpg", verbose=False)
            segments = YoloPose.from_yolo_results(results)
            ```

        Args:
            results: YOLO results from the model.

        Returns:
            YoloPose: A YOLO pose keypoints data model.
        """
        if not (summary := results[0].summary(normalize=False)):
            return YoloPose()

        cls, name, confidence, box, keypoints = [], [], [], [], []
        for res in summary:
            cls.append(res.get("class", -1))
            name.append(res.get("name", ""))
            confidence.append(res.get("confidence", -1))
            box.append(_get_box_from_yolo_result(res))
            keypoints.append(_get_keypoints_from_yolo_result(res))

        return YoloPose(
            cls=cls,
            name=name,
            confidence=confidence,
            box=box,
            keypoints=keypoints,
            orig_shape=list(results[0].orig_shape),
        )


class YoloCls(DataModel):
    """
    A class representing image classification results from a YOLO model.

    Image classification is the simplest of the three tasks and involves classifying
    an entire image into one of a set of predefined classes.

    See https://docs.ultralytics.com/tasks/classify/ for more information.

    Attributes:
        cls (list[int]): A list of classes of the detected objects, default `[]`.
        name (list[str]): A list of names of the detected objects, default `[]`.
        confidence (list[float]): A list of confidence scores of the detections,
            default `[]`.
    """

    cls: list[int] = Field(default=[])
    name: list[str] = Field(default=[])
    confidence: list[float] = Field(default=[])

    @staticmethod
    def from_yolo_results(results: list["Results"]) -> "YoloCls":
        """
        Create a YOLO classification model from the YOLO results.

        Example:
            ```python
            from ultralytics import YOLO
            from datachain.model.bbox import YoloCls

            model = YOLO("yolo11n-cls.pt")
            results = model("image.jpg", verbose=False)
            info = YoloCls.from_yolo_results(results)
            ```

        Args:
            results: YOLO results from the model.

        Returns:
            YoloCls: A YOLO classification data model.
        """
        if not results[0].probs:
            return YoloCls()

        cls, name, confidence = [], [], []
        for i, cls_id in enumerate(results[0].probs.top5):
            cls.append(cls_id)
            name.append(results[0].names[cls_id])
            confidence.append(round(results[0].probs.top5conf[i].item(), 4))

        return YoloCls(cls=cls, name=name, confidence=confidence)


def _get_box_from_yolo_result(result: dict) -> list[float]:
    """Get the bounding box coordinates from the YOLO result."""
    box = result.get("box", {})
    return [box.get(c, -1) for c in ("x1", "y1", "x2", "y2")]


def _get_obox_from_yolo_result(result: dict) -> list[float]:
    """Get the oriented bounding box coordinates from the YOLO result."""
    box = result.get("box", {})
    return [box.get(c, -1) for c in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")]


def _get_segments_from_yolo_result(result: dict) -> list[list[float]]:
    """Get the segment coordinates from the YOLO result."""
    segment = result.get("segments", {})
    return [segment.get(c, []) for c in ("x", "y")]


def _get_keypoints_from_yolo_result(result: dict) -> list[list[float]]:
    """Get the pose keypoints coordinates and visibility from the YOLO result."""
    keypoints = result.get("keypoints", {})
    return [keypoints.get(c, []) for c in ("x", "y", "visible")]
