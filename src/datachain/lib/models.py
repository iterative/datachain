from datachain.lib.data_model import DataModel


class BBox(DataModel):
    """
    A data model for representing a bounding box in image coordinates.

    Attributes:
        x1 (float): The x-coordinate of the top-left corner of the bounding box.
        y1 (float): The y-coordinate of the top-left corner of the bounding box.
        x2 (float): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (float): The y-coordinate of the bottom-right corner of the bounding box.

    The bounding box is defined by two points:
        - (x1, y1): The top-left corner of the box.
        - (x2, y2): The bottom-right corner of the box.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_coco(coco_bbox: list[float]) -> "BBox":
        """
        Converts a COCO-format bounding box to a BBox data model instance.

        Args:
            coco_bbox (list[float]): A bounding box in COCO format, represented as
                a list of four floats [x, y, width, height], where:
                - x (float): The x-coordinate of the top-left corner.
                - y (float): The y-coordinate of the top-left corner.
                - width (float): The width of the bounding box.
                - height (float): The height of the bounding box.

        Returns:
            BBox: An instance of the BBox data model.

        Notes:
            COCO (Common Objects in Context) is a dataset format for object detection,
            segmentation, and captioning.
            Find more information at https://cocodataset.org/#format-data.
        """
        x, y, w, h = coco_bbox
        return BBox(x1=x, y1=y, x2=x + w, y2=y + h)

    @staticmethod
    def from_pascal_voc(voc_bbox: list[float]) -> "BBox":
        """
        Converts a Pascal VOC-format bounding box to a BBox data model instance.

        Args:
            voc_bbox (list[float]): A bounding box in Pascal VOC format, represented as
                a list of four floats [x1, y1, x2, y2], where:
                - x1 (float): The x-coordinate of the top-left corner.
                - y1 (float): The y-coordinate of the top-left corner.
                - x2 (float): The x-coordinate of the bottom-right corner.
                - y2 (float): The y-coordinate of the bottom-right corner.

        Returns:
            BBox: An instance of the BBox data model.

        Notes:
            Pascal VOC (Visual Object Classes) is a dataset format for object detection
            and classification.
            Find more information at http://host.robots.ox.ac.uk/pascal/VOC/.
        """
        x1, y1, x2, y2 = voc_bbox
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def from_yolo(yolo_bbox: list[float], img_shape: list[int]) -> "BBox":
        """
        Converts a YOLO-format bounding box to a BBox data model instance.

        Args:
            yolo_bbox: A bounding box in YOLO format, represented as a list
                of four floats [x, y, w, h], where:
                - x (float): The x-coordinate of the center of the bounding box.
                - y (float): The y-coordinate of the center of the bounding box.
                - w (float): The width of the bounding box.
                - h (float): The height of the bounding box.
            img_shape: The shape of the image in which the bounding box is defined,
                represented as a list of two integers [width, height].

        Returns:
            BBox: An instance of the BBox data model.

        Notes:
            YOLO (You Only Look Once) is a real-time object detection system.
            Find more information at https://pjreddie.com/darknet/yolo/.
        """
        x, y, w, h = yolo_bbox
        img_w, img_h = img_shape
        return BBox(
            x1=(x - w / 2) * img_w,
            y1=(y - h / 2) * img_h,
            x2=(x + w / 2) * img_w,
            y2=(y + h / 2) * img_h,
        )


class PoseBodyPart:
    """An enumeration of body parts for pose keypoints."""

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


class Pose2D(DataModel):
    """
    A data model for representing 2D pose keypoints.

    Attributes:
        x (list[float]): The x-coordinates of the keypoints.
        y (list[float]): The y-coordinates of the keypoints.

    The keypoints are represented as lists of x and y coordinates, where each index
    corresponds to a specific body part. The body parts are defined
    by the PoseBodyPart enumeration.
    """

    x: list[float]
    y: list[float]


class Pose3D(DataModel):
    """
    A data model for representing 3D pose keypoints.

    Attributes:
        x (list[float]): The x-coordinates of the keypoints.
        y (list[float]): The y-coordinates of the keypoints.
        visible (list[float]): The visibility of the keypoints.

    The keypoints are represented as lists of x, y, and visibility values,
    where each index corresponds to a specific body part. The body parts are defined
    by the PoseBodyPart enumeration.
    """

    x: list[float]
    y: list[float]
    visible: list[float]
