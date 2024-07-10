"""
To install dependencies:

  pip install imgbeddings

"""

import uuid

from imgbeddings import imgbeddings
from PIL import Image
from sqlalchemy import tuple_

from datachain.query import C, DatasetQuery, Object, udf
from datachain.sql.functions.array import cosine_distance, euclidean_distance
from datachain.sql.types import Array, Float32


def load_image(raw):
    img = Image.open(raw)
    img.load()
    return img


@udf(
    params=(Object(load_image),),
    output={"embedding": Array(Float32)},
    method="embedding",
)
class ImageEmbeddings:
    def __init__(self):
        self.emb = imgbeddings()

    def embedding(self, img):
        emb = self.emb.to_embeddings(img)
        return (emb[0].tolist(),)


ds1_name = uuid.uuid4().hex
print(f"Saving embeddings to dataset: {ds1_name}")
# Save as a new dataset
(
    DatasetQuery(path="gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C.name.glob("*cat*.jpg"))
    .add_signals(ImageEmbeddings)
    .save(ds1_name)
)

ds2_name = uuid.uuid4().hex
source, parent, name, embedding = (
    DatasetQuery(name=ds1_name)
    .select(C.source, C.parent, C.name, C.embedding)
    .order_by(C.source, C.parent, C.name)
    .limit(1)
    .results()[0]
)
(
    DatasetQuery(name=ds1_name)
    .filter(tuple_(C.source, C.parent, C.name) != (source, parent, name))
    .mutate(
        cos_dist=cosine_distance(C.embedding, embedding),
        eucl_dist=euclidean_distance(C.embedding, embedding),
    )
    .order_by(C.cos_dist)
    .limit(10)
    .select_except(C.embedding)
    .save(ds2_name)
)

print("target:", source, parent, name, embedding[:3])
print()
print("Top 10 by cosine distance:")
for row in (
    DatasetQuery(name=ds2_name)
    .select(C.source, C.parent, C.name, C.cos_dist, C.eucl_dist)
    .order_by(C.cos_dist)
    .results()
):
    print(*row)
