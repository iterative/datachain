import json

try:
    from PIL import (
        ExifTags,
        IptcImagePlugin,
        TiffImagePlugin,
    )
except ImportError as exc:
    raise ImportError(
        "Missing dependency Pillow for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc


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
    img = file.get_value()
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
        error,
    )
