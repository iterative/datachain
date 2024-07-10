import json

from datachain.query import Object, udf
from datachain.sql.types import JSON, String

try:
    from PIL import (
        ExifTags,
        Image,
        IptcImagePlugin,
        TiffImagePlugin,
        UnidentifiedImageError,
    )
except ImportError as exc:
    raise ImportError(
        "Missing dependency Pillow for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc


def encode_image(raw):
    try:
        img = Image.open(raw)
    except UnidentifiedImageError:
        return None
    return img


@udf(
    params=(Object(encode_image),),  # Columns consumed by the UDF.
    output={
        "xmp": JSON,
        "exif": JSON,
        "iptc": JSON,
        "error": String,
    },  # Signals being returned by the UDF.
    method="image_description",
)
class GetMetadata:
    def cast(self, v):  # to JSON serializable types
        if isinstance(v, TiffImagePlugin.IFDRational):
            return float(v)
        if isinstance(v, tuple):
            return tuple(self.cast(t) for t in v)
        if isinstance(v, bytes):
            return v.decode(encoding="utf-8", errors="ignore")
        if isinstance(v, dict):
            for kk, vv in v.items():
                v[kk] = self.cast(vv)
            return v
        if isinstance(v, list):
            return [self.cast(kk) for kk in v]
        return v

    def image_description(self, img):
        (xmp, exif, iptc) = ({}, {}, {})
        if img is None:
            error = "Image format not understood"
            return ({}, {}, {}, error)
        error = ""
        xmp = img.getxmp()
        img_exif = img.getexif()
        img_iptc = IptcImagePlugin.getiptcinfo(img)

        if img_iptc:
            for k, v in img_iptc.items():
                iptc[str(k)] = self.cast(v)

        if img_exif:
            for k, v in img_exif.items():
                v = self.cast(v)
                if k in ExifTags.TAGS:
                    exif[ExifTags.TAGS[k]] = v
                if k in ExifTags.GPSTAGS:
                    exif[ExifTags.GPSTAGS[k]] = v

        return (
            json.dumps(xmp),
            json.dumps(exif),
            json.dumps(iptc),
            error,
        )
