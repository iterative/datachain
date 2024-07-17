from pydantic import BaseModel

from datachain.lib.dc import C, DataChain


class Embedding(BaseModel):
    value: float


ds_name = "feature_class"
ds = (
    DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C.name.glob("*cat*.jpg"))  # type: ignore [attr-defined]
    .limit(5)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
)

ds.save(ds_name)
