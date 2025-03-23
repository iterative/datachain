from typing import Literal, Optional

from datachain.lib.data_model import DataModel
from datachain.lib.dc import C, DataChain


class NestedFeature(DataModel):
    value: str


class Embedding(DataModel):
    value: float
    nested: NestedFeature = NestedFeature(value="nested_value")
    literal_field: Optional[Literal["end_turn", "max_tokens", "stop_sequence"]] = None


ds_name = "feature_class"
ds = (
    DataChain.from_storage("gs://dvcx-datalakes/dogs-and-cats/")
    .filter(C("file.path").glob("*cat*.jpg"))  # type: ignore [attr-defined]
    .order_by("file.path")
    .limit(5)
    .settings(cache=True, parallel=2)
    .map(emd=lambda file: Embedding(value=512), output=Embedding)
    .save(ds_name)
)

for row in ds.results():
    print(row[1])
