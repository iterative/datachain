import json

from PIL import Image
from pydantic import BaseModel

from datachain import C, DataChain, File
from datachain.sql.functions import path


class BBox(BaseModel):
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def openimage_detect(args):
    if len(args) != 2:
        raise ValueError("Group jpg-json mismatch")

    stream_jpg = args[0]
    stream_json = args[1]
    if args[0].get_file_ext() != "jpg":
        stream_jpg, stream_json = stream_json, stream_jpg

    with stream_jpg.open() as fd:
        img = Image.open(fd)

    with stream_json.open() as stream_json:
        detections = json.load(stream_json).get("detections", [])

    for i, detect in enumerate(detections):
        bbox = BBox(
            x_min=int(detect["XMin"] * img.width),
            x_max=int(detect["XMax"] * img.width),
            y_min=int(detect["YMin"] * img.height),
            y_max=int(detect["YMax"] * img.height),
        )

        fstream = File(
            source=source,
            path=f"{stream_jpg.path}/detect_{i}",
            version=stream_jpg.version,
            etag=f"{stream_jpg.etag}_{stream_jpg.etag}",
        )

        yield fstream, bbox


source = "gs://datachain-demo/openimages-v6-test-jsonpairs/"

(
    DataChain.from_storage(source)
    .filter(C("file.path").glob("*.jpg") | C("file.path").glob("*.json"))
    .agg(
        openimage_detect,
        partition_by=path.file_stem(C("file.path")),
        params=["file"],
        output={"file": File, "bbox": BBox},
    )
    .show()
)
