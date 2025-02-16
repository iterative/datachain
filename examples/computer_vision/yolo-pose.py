from ultralytics import YOLO

from datachain import DataChain, ImageFile
from datachain.model.yolo import YoloPose


def process_poses(yolo: YOLO, file: ImageFile) -> YoloPose:
    results = yolo(file.read(), verbose=False)
    return YoloPose.from_yolo_results(results)


(
    DataChain.from_storage("gs://datachain-demo/coco2017/images", type="image")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-pose.pt"))
    .map(poses=process_poses)
    .show()
)
