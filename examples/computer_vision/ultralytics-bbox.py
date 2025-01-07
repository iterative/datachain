import os

os.environ["YOLO_VERBOSE"] = "false"


from io import BytesIO

from PIL import Image
from ultralytics import YOLO

from datachain import C, DataChain, File
from datachain.model.ultralytics import YoloBBoxes


def process_bboxes(yolo: YOLO, file: File) -> YoloBBoxes:
    results = yolo(Image.open(BytesIO(file.read())))
    return YoloBBoxes.from_results(results)


(
    DataChain.from_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/")
    .filter(C("file.path").glob("*.jpg"))
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n.pt"))
    .map(boxes=process_bboxes)
    .show()
)
