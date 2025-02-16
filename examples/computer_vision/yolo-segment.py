from ultralytics import YOLO

from datachain import DataChain, File
from datachain.model.yolo import YoloSeg


def process_segments(yolo: YOLO, file: File) -> YoloSeg:
    results = yolo(file.read(), verbose=False)
    return YoloSeg.from_yolo_results(results)


(
    DataChain.from_storage("gs://datachain-demo/coco2017/images", type="image")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-seg.pt"))
    .map(segments=process_segments)
    .show()
)
