from ultralytics import YOLO

from datachain import C, DataChain, File
from datachain.model.ultralytics import YoloBBoxes


def process_bboxes(yolo: YOLO, file: File) -> YoloBBoxes:
    results = yolo(file.as_image_file().read(), verbose=False)
    return YoloBBoxes.from_results(results)


(
    DataChain.from_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/")
    .filter(C("file.path").glob("*.jpg"))
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n.pt"))
    .map(boxes=process_bboxes)
    .show()
)
