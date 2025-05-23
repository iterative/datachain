"""
To install the required dependencies:

  pip install datachain[examples]

"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import open_clip

import datachain as dc


class ImageEncoder(dc.Mapper):
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained

    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )

    def process(self, file) -> list[float]:
        img = file.read()
        img = self.preprocess(img).unsqueeze(0)
        emb = self.model.encode_image(img)
        return emb[0].tolist()


if __name__ == "__main__":
    # Run in chain
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image")
        .filter(dc.C("file.path").glob("*cat*.jpg"))
        .settings(parallel=2)
        .limit(5)
        .map(
            ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"),
            params=["file"],
            output={"emb": list[float]},
        )
        .show()
    )
