from typing import Optional

from pydantic import BaseModel

import datachain as dc
from datachain.lib.data_model import ModelStore
from datachain.lib.meta_formats import gen_datamodel_code


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
    # Dynamic JSONl schema from 2 objects
    uri = "gs://datachain-demo/jsonl/object.jsonl"
    jsonl_ds = dc.read_json(uri, format="jsonl", anon="True")
    jsonl_ds.show()

    # Dynamic JSON schema from 200 OpenImage json-pairs with validation errors
    uri = "gs://datachain-demo/openimages-v6-test-jsonpairs/*json"
    schema_uri = (
        "gs://datachain-demo/openimages-v6-test-jsonpairs/08392c290ecc9d2a.json"
    )
    json_pairs_ds = dc.read_json(
        uri, schema_from=schema_uri, jmespath="@", model_name="OpenImage", anon="True"
    )
    json_pairs_ds.show()

    uri = "gs://datachain-demo/coco2017/annotations_captions/"

    # Print JSON schema in Pydantic format from main COCO annotation
    chain = dc.read_storage(uri, anon="True").filter(dc.C("file.path").glob("*.json"))
    file = next(chain.limit(1).collect("file"))
    print(gen_datamodel_code(file, jmespath="@", model_name="Coco"))

    # Static JSON schema test parsing 3/7 objects
    static_json_ds = dc.read_json(
        uri, jmespath="licenses", spec=LicenseFeature, nrows=3, anon="True"
    )
    static_json_ds.show()

    # Dynamic JSON schema test parsing 5K objects
    dynamic_json_ds = dc.read_json(uri, jmespath="images", anon="True")
    print(dynamic_json_ds.to_pandas())

    # Static CSV with header schema test parsing 3.5K objects
    uri = "gs://datachain-demo/chatbot-csv/"
    static_csv_ds = dc.read_csv(uri, output=ChatDialog, column="chat", anon="True")
    static_csv_ds.print_schema()
    static_csv_ds.show()

    # Dynamic CSV with header schema test parsing 3/3M objects
    uri = "gs://datachain-demo/laion-aesthetics-csv/laion_aesthetics_1024_33M_1.csv"
    dynamic_csv_ds = dc.read_csv(uri, column="laion", nrows=3, anon="True")
    dynamic_csv_ds.print_schema()
    dynamic_csv_ds.show()


if __name__ == "__main__":
    main()
