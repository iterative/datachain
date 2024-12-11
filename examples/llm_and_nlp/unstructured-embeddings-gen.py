"""
To install the required dependencies:

  pip install datachain[examples]

"""

from collections.abc import Iterator

from unstructured.cleaners.core import (
    clean,
    group_broken_paragraphs,
    replace_unicode_quotes,
)
from unstructured.partition.pdf import partition_pdf
from unstructured_ingest.embed.huggingface import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingEncoder,
)

from datachain import C, DataChain, DataModel, File

source = "gs://datachain-demo/neurips/1987/"


# Define the output as a DataModel class
class Chunk(DataModel):
    key: str
    text: str
    embeddings: list[float]


# Define embedding encoder

embedding_encoder = HuggingFaceEmbeddingEncoder(config=HuggingFaceEmbeddingConfig())


# Use signatures to define UDF input/output
# these can be pydantic model or regular Python types
def process_pdf(file: File) -> Iterator[Chunk]:
    # Ingest the file
    with file.open() as f:
        chunks = partition_pdf(file=f, chunking_strategy="by_title", strategy="fast")

    # Clean the chunks and add new columns
    text_chunks = []
    for chunk in chunks:
        chunk.apply(
            lambda text: clean(
                text, bullets=True, extra_whitespace=True, trailing_punctuation=True
            )
        )
        chunk.apply(replace_unicode_quotes)
        chunk.apply(group_broken_paragraphs)
        text_chunks.append({"text": str(chunk)})

    # create embeddings
    chunks_embedded = embedding_encoder.embed_documents(text_chunks)

    # Add new rows to DataChain
    for chunk in chunks_embedded:
        yield Chunk(
            key=file.path,
            text=chunk.get("text"),
            embeddings=chunk.get("embeddings"),
        )


dc = (
    DataChain.from_storage(source)
    .settings(parallel=-1)
    .filter(C.file.path.glob("*.pdf"))
    .gen(document=process_pdf)
)

dc.save("embedded-documents")

DataChain.from_dataset("embedded-documents").show()
