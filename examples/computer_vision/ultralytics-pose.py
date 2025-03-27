from ultralytics import YOLO

import datachain as dc
from datachain.model.ultralytics import YoloPoses


def process_poses(yolo: YOLO, file: dc.File) -> YoloPoses:
    results = yolo(file.as_image_file().read(), verbose=False)
    return YoloPoses.from_results(results)


(
    dc.read_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/")
    .filter(dc.C("file.path").glob("*.jpg"))
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-pose.pt"))
    .map(poses=process_poses)
    .show()
)
