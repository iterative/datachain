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

from .bbox import YoloBBox, YoloBBoxes, YoloOBBox, YoloOBBoxes
from .pose import YoloPose, YoloPoses
from .segment import YoloSegment, YoloSegments

__all__ = [
    "YoloBBox",
    "YoloBBoxes",
    "YoloOBBox",
    "YoloOBBoxes",
    "YoloPose",
    "YoloPoses",
    "YoloSegment",
    "YoloSegments",
]
