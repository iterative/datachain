"""
To install the required dependencies:

  pip install datachain[examples]

"""

import os

import open_clip

import datachain as dc
from datachain import C, File

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class ClipImageEncoder:
    def __init__(self, model_name: str, pretrained: str):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )


def embeddings(file: File, encoder: ClipImageEncoder) -> list[float]:
    img = file.read()
    img = encoder.preprocess(img).unsqueeze(0)
    emb = encoder.model.encode_image(img)
    return emb[0].tolist()


if __name__ == "__main__":
    (
        dc.read_storage("gs://datachain-demo/dogs-and-cats/", type="image", anon=True)
        .filter(C("file.path").glob("*cat*.jpg"))
        .limit(5)
        .settings(parallel=True)
        .setup(encoder=lambda: ClipImageEncoder("ViT-B-32", "laion2b_s34b_b79k"))
        .map(emb=embeddings)
        .show()
    )
