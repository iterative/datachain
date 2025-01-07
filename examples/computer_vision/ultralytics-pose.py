import os

os.environ["YOLO_VERBOSE"] = "false"


from io import BytesIO

from PIL import Image
from ultralytics import YOLO

from datachain import C, DataChain, File
from datachain.model.ultralytics import YoloPoses


def process_poses(yolo: YOLO, file: File) -> YoloPoses:
    results = yolo(Image.open(BytesIO(file.read())))
    return YoloPoses.from_results(results)


(
    DataChain.from_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/")
    .filter(C("file.path").glob("*.jpg"))
    .limit(20)
    .setup(yolo=lambda: YOLO("yolo11n-pose.pt"))
    .map(poses=process_poses)
    .show()
)
