#
# pip install unstructured[all-docs]
# libmagic
#
# partition_object supports via unstructured library:
#
# "csv", "doc", "docx", "epub", "image", "md", "msg", "odt", "org",
# "pdf", "ppt", "pptx", "rtf", "rst", "tsv", "xlsx"

from transformers import pipeline
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dataframe

from datachain import C, DataChain

device = "cpu"
model = "pszemraj/led-large-book-summary"
source = "gs://datachain-demo/nlp-infobooks/"


def partition_object(file):
    with file.open() as raw:
        elements = partition(file=raw, metadata_filename=file.name)
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


ds = (
    DataChain.from_storage(source)
    .filter(C("file.name").glob("*.pdf"))
    .limit(1)
    .map(
        partition_object,
        params=["file"],
        output={"elements": dict, "title": str, "text": str, "error": str},
    )
)

ds = ds.map(cleanse, output={"clean": str})
ds = ds.map(summarize, output={"summary": str})
results = ds.select("text", "summary").collect()

for story in results:
    print("\n *********** the original: ********** ")
    print(story[0])

    print("\n *********** the summary: *********** ")
    print(story[1])
    print("\n")
