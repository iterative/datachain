import os
from io import BytesIO

from numpy import asarray
from PIL import Image
from ultralytics import YOLO

from datachain import DataChain, File
from datachain.model.ultralytics import YoloBBoxes
from datachain.toolkit.ultralytics import visualize_yolo

OUTPUT_DIR = "output/bbox"


def process_bboxes(yolo: YOLO, file: File) -> YoloBBoxes:
    # read image
    img = Image.open(BytesIO(file.read()))

    # detect objects using YOLO model
    results = yolo(img, verbose=False)
    # convert results to YoloBBoxes signal
    signal = YoloBBoxes.from_results(results)

    # visualize results
    img2 = visualize_yolo(asarray(img), signal)
    img2.save(f"{OUTPUT_DIR}/{file.get_file_stem()}.jpg")

    return signal


os.makedirs(OUTPUT_DIR, exist_ok=True)

(
    DataChain.from_storage("gs://datachain-demo/coco2017/images")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n.pt"))
    .map(boxes=process_bboxes)
    .show()
)
