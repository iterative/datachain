"""
To install dependencies:

  pip install open_clip_torch

"""

import open_clip

from datachain import C, DataChain, Mapper


class ImageEncoder(Mapper):
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
        DataChain.from_storage("gs://datachain-demo/dogs-and-cats/", type="image")
        .filter(C("file.path").glob("*cat*.jpg"))
        .settings(parallel=2)
        .limit(5)
        .map(
            ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"),
            params=["file"],
            output={"emb": list[float]},
        )
        .show()
    )
