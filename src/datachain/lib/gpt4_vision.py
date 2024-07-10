import base64
import io
import os

import requests

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError as exc:
    raise ImportError(
        "Missing dependency Pillow for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc

from datachain.query import Object, udf
from datachain.sql.types import String

DEFAULT_FIT_BOX = (500, 500)
DEFAULT_TOKENS = 300


def encode_image(raw):
    try:
        img = Image.open(raw)
    except UnidentifiedImageError:
        return None
    img.load()
    img = ImageOps.fit(img, DEFAULT_FIT_BOX)
    output = io.BytesIO()
    img.save(output, format="JPEG")
    hex_data = output.getvalue()
    return base64.b64encode(hex_data).decode("utf-8")


@udf(
    params=(Object(encode_image),),  # Columns consumed by the UDF.
    output={
        "description": String,
        "error": String,
    },  # Signals being returned by the UDF.
    method="image_description",
)
class DescribeImage:
    def __init__(
        self,
        prompt="What is in this image?",
        max_tokens=DEFAULT_TOKENS,
        key="",
        timeout=30,
    ):
        if not key:
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise ValueError(
                    "No key found. Please pass key or set the OPENAI_API_KEY "
                    "environment variable."
                )
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        self.timeout = timeout

    def image_description(self, base64_image):
        if base64_image is None:
            return ("", "Unknown image format")

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        json_response = response.json()

        if "error" in json_response:
            error = str(json_response["error"])
            openai_description = ""
        else:
            error = ""
            openai_description = json_response["choices"][0]["message"]["content"]

        return (openai_description, error)
