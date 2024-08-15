# pip install defusedxml
import json

from PIL import (
    ExifTags,
    IptcImagePlugin,
    TiffImagePlugin,
)

from datachain import C, DataChain

source = "gs://datachain-demo/open-images-v6/"


def cast(v):  # to JSON serializable types
    if isinstance(v, TiffImagePlugin.IFDRational):
        return float(v)
    if isinstance(v, tuple):
        return tuple(cast(t) for t in v)
    if isinstance(v, bytes):
        return v.decode(encoding="utf-8", errors="ignore")
    if isinstance(v, dict):
        for kk, vv in v.items():
            v[kk] = cast(vv)
        return v
    if isinstance(v, list):
        return [cast(kk) for kk in v]
    return v


def image_description(file):
    (xmp, exif, iptc) = ({}, {}, {})
    try:
        img = file.read()
        xmp = img.getxmp()
        img_exif = img.getexif()
        img_iptc = IptcImagePlugin.getiptcinfo(img)
    except Exception as err:  # noqa: BLE001
        error = str(err)
        return ({}, {}, {}, error)

    if img_iptc:
        for k, v in img_iptc.items():
            iptc[str(k)] = cast(v)

    if img_exif:
        for k, v in img_exif.items():
            v = cast(v)
            if k in ExifTags.TAGS:
                exif[ExifTags.TAGS[k]] = v
            if k in ExifTags.GPSTAGS:
                exif[ExifTags.GPSTAGS[k]] = v

    return (
        json.dumps(xmp),
        json.dumps(exif),
        json.dumps(iptc),
        "",
    )


if __name__ == "__main__":
    (
        DataChain.from_storage(source, type="image")
        .settings(parallel=-1)
        .filter(C("file.path").glob("*.jpg"))
        .limit(5000)
        .map(
            image_description,
            params=["file"],
            output={"xmp": dict, "exif": dict, "iptc": dict, "error": str},
        )
        .select("file.path", "xmp", "exif", "iptc", "error")
        .filter((C("xmp") != "{}") | (C("exif") != "{}") | (C("iptc") != "{}"))
        .show()
    )
