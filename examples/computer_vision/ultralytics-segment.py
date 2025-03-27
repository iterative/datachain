from ultralytics import YOLO

import datachain as dc
from datachain.model.ultralytics import YoloSegments


def process_segments(yolo: YOLO, file: dc.File) -> YoloSegments:
    results = yolo(file.as_image_file().read(), verbose=False)
    return YoloSegments.from_results(results)


(
    dc.read_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/")
    .filter(dc.C("file.path").glob("*.jpg"))
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-seg.pt"))
    .map(segments=process_segments)
    .show()
)
