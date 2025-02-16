from ultralytics import YOLO

from datachain import DataChain, ImageFile
from datachain.model.yolo import YoloBox


def process_boxes(yolo: YOLO, file: ImageFile) -> YoloBox:
    results = yolo(file.read(), verbose=False)
    return YoloBox.from_yolo_results(results)


(
    DataChain.from_storage("gs://datachain-demo/coco2017/images", type="image")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n.pt"))
    .map(boxes=process_boxes)
    .show()
)
