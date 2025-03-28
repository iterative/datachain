import base64
import os

import requests

import datachain as dc

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SOURCE = "gs://datachain-demo/dogs-and-cats/"
DEFAULT_FIT_BOX = (500, 500)
DEFAULT_TOKENS = 300


def describe_image(
    file,
    model="gpt-4o",
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    base64_image = base64.b64encode(file.read()).decode("utf-8")
    if base64_image is None:
        return ("", "Unknown image format")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    json_response = response.json()

    if "error" in json_response:
        error = str(json_response["error"])
        openai_description = ""
    else:
        error = ""
        openai_description = json_response["choices"][0]["message"]["content"]

    return (openai_description, error)


if __name__ == "__main__":
    (
        dc.read_storage(
            SOURCE,
            anon=True,
        )
        .filter(dc.C("file.path").glob("*cat*.jpg"))
        .limit(10)
        .map(
            lambda file: describe_image(
                file,
                key=OPENAI_API_KEY,
                max_tokens=300,
                prompt="What is in this image?",
            ),
            params=["file"],
            output={"description": str, "error": str},
        )
        .select("file.source", "file.path", "description", "error")
        .show()
    )
