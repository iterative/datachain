"""
fashion-clip UDF, using the
[fashion-clip](https://pypi.org/project/fashion-clip/#description) package.

Generated embeddings are stored json-encoded.

To install script dependencies: pip install tabulate fashion-clip
"""

import json

from fashion_clip.fashion_clip import FashionCLIP
from sqlalchemy import JSON
from tabulate import tabulate

from datachain.lib.param import Image
from datachain.query import C, DatasetQuery, udf


@udf(
    params=(Image(),),
    output={"fclip": JSON},
    method="fashion_clip",
    batch=10,
)
class MyFashionClip:
    def __init__(self):
        self.fclip = FashionCLIP("fashion-clip")

    def fashion_clip(self, inputs):
        embeddings = self.fclip.encode_images(
            [input[0] for input in inputs], batch_size=1
        )
        return [(json.dumps(emb),) for emb in embeddings.tolist()]


if __name__ == "__main__":
    # This example processes 5 objects in the new dataset and generates the
    # embeddings for them.
    DatasetQuery(path="gs://dvcx-zalando-hd-resized/zalando-hd-resized/").filter(
        C.name.glob("*.jpg")
    ).limit(5).add_signals(MyFashionClip).save("zalando_hd_emb")

    print(tabulate(DatasetQuery(name="zalando_hd_emb").results()[:5]))
