import json

from PIL import Image

import datachain as dc
from datachain import File, model
from datachain.func import path


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
        bbox = model.BBox.from_albumentations(
            [detect[k] for k in ("XMin", "YMin", "XMax", "YMax")],
            img_size=(img.width, img.height),
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
    dc.from_storage(source)
    .filter(dc.C("file.path").glob("*.jpg") | dc.C("file.path").glob("*.json"))
    .agg(
        openimage_detect,
        partition_by=path.file_stem("file.path"),
        params=["file"],
        output={"file": File, "bbox": model.BBox},
    )
    .show()
)
