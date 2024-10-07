# Set logger to debug level
import logging

from pydantic import BaseModel

from datachain.lib.dc import C, DataChain

logging.basicConfig(level=logging.INFO)


class Embedding(BaseModel):
    value: float


ds_name = "feature_class_error"
ds = (
    DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C("file.path").glob("*cat*.jpg"))
    .limit(5)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
)
ds.select("file.path", "emd.value").show(limit=5, flatten=True)
ds.save(ds_name)
raise Exception("This is a test exception")
