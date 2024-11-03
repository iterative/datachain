from datachain.lib.data_model import DataModel


class BoundingBox(DataModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_coco(coco_bbox: list[float]) -> "BoundingBox":
        x, y, w, h = coco_bbox
        return BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)

    @staticmethod
    def from_pascal_voc(voc_bbox: list[float]) -> "BoundingBox":
        x1, y1, x2, y2 = voc_bbox
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def from_yolo(yolo_bbox: list[float], img_shape: list[int]) -> "BoundingBox":
        x, y, w, h = yolo_bbox
        img_h, img_w = img_shape
        return BoundingBox(
            x1=(x - w / 2) * img_w,
            y1=(y - h / 2) * img_h,
            x2=(x + w / 2) * img_w,
            y2=(y + h / 2) * img_h,
        )


class Pose2D(DataModel):
    x: list[float]
    y: list[float]


class Pose3D(DataModel):
    x: list[float]
    y: list[float]
    visible: list[float]
