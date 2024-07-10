#
# TODO:
# refactor lib/meta_formats/read_scema into a Datachain method
#
# ER: add support for Optional fields in read_schema()
# ER: add support for headless CSV within static schema only
# ER: fix the bug in datamodel-codegen failing to recognize csv float and int columns
#
# Open issues:
# 1. A single filename cannot be passed as schema source (#1563)
# 2. Need syntax like "file.open(encoding='utf-8')" to avoid "type=text" (#1614)
# 3. Need syntax like "datachain.collate(func -> Any)" (#1615)
# 4. "Feature" does not tolerate creating a class twice (#1617)
# 5. Unsure how to deal with 'folder' pseudo-files in cloud systems(#1618)
# 6. There should be exec() method to force-run the existing chain (#1616)
# 7. data-model-codegenerator: datamodel-codegen reports all CSV fields as 'str'.
# 8. from_json and from_csv methods do not filter empty files from AWS
# dependencies:
# pip install datamodel-code-generator
# pip install jmespath

from typing import Optional

from pydantic import BaseModel

from datachain.lib.dc import C, DataChain
from datachain.lib.feature_utils import pydantic_to_feature


# Sample model for static JSON model
class LicenseModel(BaseModel):
    url: str
    id: int
    name: str


LicenseFeature = pydantic_to_feature(LicenseModel)


# Sample model for static CSV model
class ChatDialog(BaseModel):
    id: Optional[int] = None
    count: Optional[int] = None
    sender: Optional[str] = None
    text: Optional[str] = None


ChatFeature = pydantic_to_feature(ChatDialog)


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
    # print(json_pairs_ds.collect()[0])

    uri = "gs://datachain-demo/coco2017/annotations_captions/"

    print()
    print("========================================================================")
    print("Reading JSON schema from main COCO annotation")
    print("========================================================================")
    chain = (
        DataChain.from_storage(uri)
        .filter(C.name.glob("*.json"))
        .show_json_schema(jmespath="@", model_name="Coco")
    )
    chain.save()

    print()
    print("========================================================================")
    print("static JSON schema test parsing 7 objects")
    print("========================================================================")
    static_json_ds = DataChain.from_json(uri, jmespath="licenses", spec=LicenseFeature)
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
    static_csv_ds = DataChain.from_csv(uri, spec=ChatFeature)
    print(static_csv_ds.to_pandas())

    uri = "gs://datachain-demo/laion-aesthetics-csv"
    print()
    print("========================================================================")
    print("dynamic CSV with header schema test parsing 3M objects")
    print("========================================================================")
    dynamic_csv_ds = DataChain.from_csv(uri, object_name="laion", show_schema=True)
    print(dynamic_csv_ds.to_pandas())


if __name__ == "__main__":
    main()
