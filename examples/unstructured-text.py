#
# pip install unstructured[all-docs]
# libmagic
#
# partition_object supports via unstructured library:
#
# "csv", "doc", "docx", "epub", "image", "md", "msg", "odt", "org",
# "pdf", "ppt", "pptx", "rtf", "rst", "tsv", "xlsx"

from transformers import pipeline

from datachain.lib.dc import DataChain
from datachain.lib.unstructured import PartitionObject
from datachain.query import C
from datachain.sql.types import String

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
    DataChain(
        source,
        anon=True,
    )
    .filter(C.name.glob("*.pdf"))
    .limit(1)
    .map(PartitionObject(), parallel=False)
)

ds = ds.map(cleanse, output={"clean": String})
ds = ds.map(summarize, output={"summary": String})
results = ds.select("text", "summary").results()

for story in results:
    print("\n *********** the original: ********** ")
    print(story[0])

    print("\n *********** the summary: *********** ")
    print(story[1])
    print("\n")
