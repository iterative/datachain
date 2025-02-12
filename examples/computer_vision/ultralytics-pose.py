import os
from io import BytesIO

from numpy import asarray
from PIL import Image
from ultralytics import YOLO

from datachain import DataChain, File
from datachain.model.ultralytics import YoloPoses
from datachain.toolkit.ultralytics import visualize_yolo

OUTPUT_DIR = "output/pose"


def process_poses(yolo: YOLO, file: File) -> YoloPoses:
    # read image
    img = Image.open(BytesIO(file.read()))

    # detect objects using YOLO model
    results = yolo(img, verbose=False)
    # convert results to YoloPoses signal
    signal = YoloPoses.from_results(results)

    # visualize results
    img2 = visualize_yolo(asarray(img), signal)
    img2.save(f"{OUTPUT_DIR}/{file.get_file_stem()}.jpg")

    return signal


os.makedirs(OUTPUT_DIR, exist_ok=True)

(
    DataChain.from_storage("gs://datachain-demo/coco2017/images")
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-pose.pt"))
    .map(poses=process_poses)
    .show()
)
