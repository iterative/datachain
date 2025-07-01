"""
To install the required dependencies:

  pip install datachain[examples]

"""

from PIL import (
    ExifTags,
    IptcImagePlugin,
    TiffImagePlugin,
)

import datachain as dc
from datachain import C, DataModel, File


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


class ImageDescription(DataModel):
    xmp: dict
    exif: dict
    iptc: dict


def image_description(file: File) -> tuple[ImageDescription, str]:
    xmp, exif, iptc = {}, {}, {}
    try:
        img = file.read()
        xmp = img.getxmp()
        img_exif = img.getexif()
        img_iptc = IptcImagePlugin.getiptcinfo(img)
    except Exception as err:  # noqa: BLE001
        return ImageDescription(xmp={}, exif={}, iptc={}), str(err)

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

    return (ImageDescription(xmp=xmp, exif=exif, iptc=iptc), "")


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/open-images-v6/", type="image", anon=True)
        .filter(C("file.path").glob("*.jpg"))
        .limit(5000)
        .settings(parallel=True)
        .map(image_description, output=("description", "error"))
        .filter(
            (C("description.xmp") != "{}")
            | (C("description.exif") != "{}")
            | (C("description.iptc") != "{}")
        )
        .show()
    )
