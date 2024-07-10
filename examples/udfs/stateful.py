"""
To install dependencies:

  pip install open_clip_torch

"""

import uuid

import open_clip

from datachain.lib.dc import C, DataChain
from datachain.lib.image import ImageFile

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)


def encode_image(file: ImageFile) -> list[float]:
    img = file.get_value()
    img = preprocess(img).unsqueeze(0)
    emb = model.encode_image(img)
    return emb[0].tolist()


if __name__ == "__main__":
    ds_name = uuid.uuid4().hex
    print(f"Saving to dataset: {ds_name}")
    # Save as a new dataset
    (
        DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/", type="image")
        .settings(parallel=2)
        .filter(C.name.glob("*cat*.jpg"))
        .limit(5)
        .map(emb=encode_image)
        .save(ds_name)
    )

    for row in DataChain.from_dataset(ds_name).results()[:2]:
        print("default columns: ", row[:-1])
        print("embedding[:10]:  ", row[-1][:10])
        print(f"type: {type(row[-1]).__name__}, len: {len(row[-1])}")
        print()
