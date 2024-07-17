from pydantic import BaseModel

from datachain.lib.dc import C, DataChain
from datachain.query.dataset import DatasetQuery


class Embedding(BaseModel):
    value: float


ds = (
    DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C.name.glob("*cat*.jpg"))  # type: ignore [attr-defined]
    .limit(5)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
)

ds.save("parallel_index")


for row in DatasetQuery(name="parallel_index").select(C.name).results():
    print(row[0])
