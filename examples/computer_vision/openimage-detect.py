import json
from collections.abc import Iterator

from PIL import Image

import datachain as dc
from datachain import C, File, model
from datachain.func import path


# Example showing extraction of bounding boxes from Open Images dataset
# that comes as pairs of JPG and JSON files.
def openimage_detect(file: list[File]) -> Iterator[tuple[File, model.BBox]]:
    if len(file) != 2:
        raise ValueError("Group jpg-json mismatch")

    stream_jpg = file[0]
    stream_json = file[1]
    source = stream_jpg.source
    if stream_jpg.get_file_ext() != "jpg":
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


(
    dc.read_storage("gs://datachain-demo/openimages-v6-test-jsonpairs/", anon=True)
    .filter(C("file.path").glob("*.jpg") | C("file.path").glob("*.json"))
    .settings(cache=True, parallel=True)
    .agg(
        openimage_detect,
        partition_by=path.file_stem("file.path"),
        output=("file", "bbox"),
    )
    .show()
)
