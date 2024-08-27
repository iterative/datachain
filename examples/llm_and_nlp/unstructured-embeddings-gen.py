from collections.abc import Iterator
from typing import List

from unstructured.cleaners.core import (
    clean,
    group_broken_paragraphs,
    replace_unicode_quotes,
)
from unstructured.embed.huggingface import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingEncoder,
)
from unstructured.partition.pdf import partition_pdf

from datachain import C, DataChain, DataModel, File


# Define the output as a DataModel class
class Chunk(DataModel):
    key: str
    text: str
    summary: str
    embeddings: List[float]


# Define embedding encoder

embedding_encoder = HuggingFaceEmbeddingEncoder(config=HuggingFaceEmbeddingConfig())


# Use signatures to define UDF input/output (these can be pydantic model or regular Python types)
def process_pdf(file: File) -> Iterator[Chunk]:
    # Ingest the file
    with file.open() as f:
        chunks = partition_pdf(file=f, chunking_strategy="by_title", strategy="fast")

    title = str(chunks[0])

    # Clean the chunks and add new columns
    for chunk in chunks:
        chunk.apply(
            lambda text: clean(
                text, bullets=True, extra_whitespace=True, trailing_punctuation=True
            )
        )
        chunk.apply(replace_unicode_quotes)
        chunk.apply(group_broken_paragraphs)

    cleaned_text = "\n\n".join([str(el) for el in chunks])
    helper = pipeline(model=model, device=device)
    summary = helper(clean, max_length=200)[0]["summary_text"]

    # create embeddings
    chunks_embedded = embedding_encoder.embed_documents(chunks)

    # Add new rows to DataChain
    for chunk in chunks_embedded:
        yield Chunk(
            key=file.path,
            text=chunk.text,
            summary=summary,
            embeddings=chunk.embeddings,
        )


dc = (
    DataChain.from_storage(source)
    .settings(parallel=-1)
    .filter(C.file.path.glob("*.pdf"))
    .gen(document=process_pdf)
)

dc.save("embedded-documents")

DataChain.from_dataset("embedded-documents").show()
