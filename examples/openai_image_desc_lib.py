# pip install Pillow

import os

from datachain.lib.dc import C, DataChain
from datachain.lib.gpt4_vision import describe_image

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

source = "gs://dvcx-datalakes/dogs-and-cats/"

if __name__ == "__main__":
    results = (
        DataChain.from_storage(
            source,
            anon=True,
        )
        .filter(C("name").glob("cat*.jpg"))
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
        .select("file.source", "file.parent", "file.name", "description", "error")
        .results()
    )
    print(*results, sep="\n")
