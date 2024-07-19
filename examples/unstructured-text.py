#
# pip install unstructured[all-docs]
# libmagic
#
# partition_object supports via unstructured library:
#
# "csv", "doc", "docx", "epub", "image", "md", "msg", "odt", "org",
# "pdf", "ppt", "pptx", "rtf", "rst", "tsv", "xlsx"

from transformers import pipeline

from datachain.lib.dc import C, DataChain
from datachain.lib.unstructured import partition_object

device = "cpu"
model = "pszemraj/led-large-book-summary"
source = "gs://dvcx-datalakes/NLP/infobooks/"


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
    .filter(C("name").glob("*.pdf"))
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
