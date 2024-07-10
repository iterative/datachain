import os

from langchain_community.embeddings import OpenAIEmbeddings
from parse import parse
from text_loaders import load_bibtex, pdf_pages

from datachain.query import C, DatasetQuery, Object, udf
from datachain.sql.types import Array, Float, String

# This will only run on 1/512th of the data! Change to 1 to run on all the data.
PDF_SUBSET_MOD = 512

BASE_URL = "https://proceedings.neurips.cc/paper_files/paper/"


def chomp(ss, prefix):
    assert ss.startswith(prefix)
    return ss[len(prefix) :]


@udf(
    params=(Object(load_bibtex),),
    output={
        "bib_parent": String,
        "bib_name": String,
        "bib_title": String,
        "bib_authors": Array(String),
    },
)
def get_bib_data(bib_data):
    (entry,) = bib_data.entries.keys()  # should only be one entry
    data = bib_data.entries[entry]

    url = data.fields["url"]
    url = chomp(url, BASE_URL)
    parent, name = os.path.split(url)

    title = data.fields.get("title", None)

    authors = [str(author) for author in data.persons.get("author", [])]
    return (parent, name, title, authors)


@udf(
    params=("bib_parent", "bib_name"),
    output={"hash_": String, "key": String},
)
def parse_bib_filename(parent, paper_filename):
    parse_result = parse("{}-{}.{}", paper_filename)
    hash_, _, _ = parse_result
    key_ = f"{parent}-{hash_}"
    return (hash_, key_)


@udf(
    params=("parent", "name"),
    output={
        "hash_": String,
        "object_type": String,
        "object_subtype": String,
        "ext": String,
        "key": String,
    },
)
def parse_pdf_filename(parent, filename):
    parse_result = parse("{}-{}.{}", filename)
    hash_, object_type, ext = parse_result

    # Ensure consistent camel case
    object_type = object_type.replace(" ", "").replace("_", "")

    split_list = object_type.split("-")
    assert len(split_list) <= 2
    object_type = split_list[0]
    object_subtype = split_list[1] if len(split_list) > 1 else None

    key_ = f"{parent}-{hash_}"
    return hash_, object_type, object_subtype, ext, key_


@udf(
    params=("page",),
    output={"embed": Array(Float)},
    batch=64,
)
def embed_page(pages):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    assert openai_api_key.startswith("sk-")
    openai = OpenAIEmbeddings(openai_api_key=openai_api_key)
    del openai_api_key

    embed = openai.embed_documents([page[0] for page in pages])
    return [(ee,) for ee in embed]


# Remove last line to include post-2000 papers
subset_ds = (
    DatasetQuery("s3://neurips-papers/")
    .filter(~C.name.glob("*.zip"))
    .filter(~(C.parent.glob("20*/file") | C.parent.glob("20*/hash")))
)

pdf_ds = (
    subset_ds.filter(C.parent != "")
    .filter(C.name.glob("*.pdf"))
    .filter(C.random % PDF_SUBSET_MOD == 0)
    .generate(pdf_pages)
    .add_signals(parse_pdf_filename)
    .add_signals(embed_page)
)

bib_ds = (
    subset_ds.filter(C.parent != "")
    .filter((C.name == "bibtex") | C.name.glob("*.bib"))
    .add_signals(get_bib_data)
    .add_signals(parse_bib_filename)
)

pdf_ds.join(bib_ds, predicates="key")
