#
# pip install unstructured[pdf] huggingface_hub[hf_transfer]
#
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from transformers import pipeline
from unstructured.partition.pdf import PartitionStrategy
from unstructured.partition.pdf import partition_pdf as partition
from unstructured.staging.base import convert_to_dataframe

from datachain import DataChain

device = "cpu"
model = "pszemraj/led-large-book-summary"
source = "gs://datachain-demo/nlp-infobooks/*.pdf"


def partition_object(file):
    with file.open() as raw:
        elements = partition(
            file=raw, metadata_filename=file.name, strategy=PartitionStrategy.FAST
        )
    title = str(elements[0])
    text = "\n\n".join([str(el) for el in elements])
    df = convert_to_dataframe(elements)
    return (df.to_json(), title, text, "")


def cleanse(text):
    separator = "Listen to this story"
    head, _sep, _tail = text.partition(separator)
    return (head,)


def summarize(clean):
    helper = pipeline(model=model, device=device)
    summary = helper(clean, max_length=200)[0]["summary_text"]
    return (summary,)


dc = (
    DataChain.from_storage(source)
    .limit(1)
    .map(
        partition_object,
        params=["file"],
        output={"elements": dict, "title": str, "text": str, "error": str},
    )
)

dc = dc.map(cleanse, output={"clean": str})
dc = dc.map(summarize, output={"summary": str})
results = dc.select("text", "summary").collect()

for story in results:
    print("\n *********** the original: ********** ")
    print(story[0])

    print("\n *********** the summary: *********** ")
    print(story[1])
    print("\n")
