from pydantic import BaseModel

import datachain as dc


class Embedding(BaseModel):
    value: float


ds_name = "feature_class"
ds = (
    dc.read_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(dc.C("file.path").glob("*cat*.jpg"))
    .order_by("file.path")
    .limit(5)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
)
ds.select("file.path", "emd.value").show(limit=5, flatten=True)
ds.save(ds_name)
