# pip install datamodel-code-generator jmespath

from typing import Optional

from pydantic import BaseModel

from datachain import C, DataChain
from datachain.lib.data_model import ModelStore


# Sample model for static JSON model
class LicenseModel(BaseModel):
    url: str
    id: int
    name: str


LicenseFeature = ModelStore.register(LicenseModel)


# Sample model for static CSV model
class ChatDialog(BaseModel):
    id: Optional[int] = None
    count: Optional[int] = None
    sender: Optional[str] = None
    text: Optional[str] = None


ChatFeature = ModelStore.register(ChatDialog)


def main():
    print()
    print("========================================================================")
    print("Dynamic JSONl schema from 2 objects")
    print("========================================================================")
    uri = "gs://datachain-demo/jsonl/object.jsonl"
    jsonl_ds = DataChain.from_json(uri, meta_type="jsonl", show_schema=True)
    print(jsonl_ds.to_pandas())

    print()
    print("========================================================================")
    print("Dynamic JSON schema from 200 OpenImage json-pairs with validation errors")
    print("========================================================================")
    uri = "gs://datachain-demo/openimages-v6-test-jsonpairs/*json"
    schema_uri = (
        "gs://datachain-demo/openimages-v6-test-jsonpairs/08392c290ecc9d2a.json"
    )
    json_pairs_ds = DataChain.from_json(
        uri, schema_from=schema_uri, jmespath="@", model_name="OpenImage"
    )
    print(json_pairs_ds.to_pandas())
    # print(list(json_pairs_ds.collect())[0])

    uri = "gs://datachain-demo/coco2017/annotations_captions/"

    print()
    print("========================================================================")
    print("Reading JSON schema from main COCO annotation")
    print("========================================================================")
    chain = (
        DataChain.from_storage(uri)
        .filter(C("file.name").glob("*.json"))
        .show_json_schema(jmespath="@", model_name="Coco")
    )
    chain.save()

    print()
    print("========================================================================")
    print("static JSON schema test parsing 3/7 objects")
    print("========================================================================")
    static_json_ds = DataChain.from_json(
        uri, jmespath="licenses", spec=LicenseFeature, nrows=3
    )
    print(static_json_ds.to_pandas())

    print()
    print("========================================================================")
    print("dynamic JSON schema test parsing 5K objects")
    print("========================================================================")
    dynamic_json_ds = DataChain.from_json(uri, jmespath="images", show_schema=True)
    print(dynamic_json_ds.to_pandas())

    uri = "gs://datachain-demo/chatbot-csv/"
    print()
    print("========================================================================")
    print("static CSV with header schema test parsing 3.5K objects")
    print("========================================================================")
    static_csv_ds = DataChain.from_csv(uri, output=ChatDialog, object_name="chat")
    static_csv_ds.print_schema()
    print(static_csv_ds.to_pandas())

    uri = "gs://datachain-demo/laion-aesthetics-csv"
    print()
    print("========================================================================")
    print("dynamic CSV with header schema test parsing 3/3M objects")
    print("========================================================================")
    dynamic_csv_ds = DataChain.from_csv(uri, object_name="laion", nrows=3)
    dynamic_csv_ds.print_schema()
    print(dynamic_csv_ds.to_pandas())


if __name__ == "__main__":
    main()
