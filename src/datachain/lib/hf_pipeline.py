import json

from transformers import pipeline

from datachain.query import Object, udf
from datachain.sql.types import JSON, String

try:
    from PIL import (
        Image,
        UnidentifiedImageError,
    )
except ImportError as exc:
    raise ImportError(
        "Missing dependency Pillow for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc


def read_image(raw):
    try:
        img = Image.open(raw)
    except UnidentifiedImageError:
        return None
    img.load()
    return img.convert("RGB")


def read_object(raw):
    return raw.read()


def read_text(raw):
    return read_object(raw).decode("utf-8")


@udf(
    params=(Object(read_image),),  # Columns consumed by the UDF.
    output={
        "model_output": JSON,
        "error": String,
    },  # Signals being returned by the UDF.
    method="image_processor",
)
class ImageHelper:
    def __init__(self, model, device, **kwargs):
        self.helper = pipeline(model=model, device=device)
        self.kwargs = kwargs

    def image_processor(self, imgs):
        result = self.helper(
            imgs,
            **self.kwargs,
        )
        return (json.dumps(result), "")


@udf(
    params=(Object(read_text),),  # Columns consumed by the UDF.
    output={
        "model_output": JSON,
        "error": String,
    },  # Signals being returned by the UDF.
    method="text_processor",
)
class TextHelper:
    def __init__(self, model, device, **kwargs):
        self.helper = pipeline(model=model, device=device)
        self.kwargs = kwargs

    def text_processor(self, text):
        result = self.helper(
            text,
            **self.kwargs,
        )
        return (json.dumps(result), "")


@udf(
    params=(Object(read_object),),  # Columns consumed by the UDF.
    output={
        "model_output": JSON,
        "error": String,
    },  # Signals being returned by the UDF.
    method="raw_processor",
)
class RawHelper:
    def __init__(self, model, device, **kwargs):
        self.helper = pipeline(model=model, device=device)
        self.kwargs = kwargs

    def raw_processor(self, obj):
        result = self.helper(
            obj,
            **self.kwargs,
        )
        return (json.dumps(result), "")
