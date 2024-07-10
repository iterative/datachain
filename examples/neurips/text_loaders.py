import logging
import re

import PyPDF2
from pybtex.database.input import bibtex

from datachain.query import DatasetRow, Object, udf
from datachain.sql.types import Int, String

logger = logging.getLogger("datachain")


def sanitize_utf8(original):
    """Clean up invalid UTF characters that may cause failures when writing to a DB."""
    return (
        original.encode("utf-8", "ignore").decode("utf-8", "ignore").replace("\x00", "")
    )


def sanitize_ascii(original):
    """More restrictive than sanitize_utf8 when we want just basic keyboard chars.
    This also puts all the text on a single line by eliminating all line breaks.
    If we expect a large amount of non-English text, then it makes sense to first use
    the Unidecode package.
    """
    sanitized = (
        original.encode("ascii", "ignore").decode("ascii", "ignore").replace("\x00", "")
    )
    sanitized = " ".join(sanitized.splitlines())
    sanitized = re.sub(r"[^\x20-\x7e]", r"", sanitized)
    # These should always pass after the previous conversions
    assert isinstance(sanitized, str)
    assert sanitized.isascii()
    assert sanitized.isprintable()
    return sanitized


def compress_whitespace(original):
    """PDF parsers often result in some weird whitespace."""
    return " ".join(original.split())


def load_pdf_pages(raw):
    raw_name = raw.info()["name"]

    pages = []
    try:
        pdf_reader = PyPDF2.PdfReader(raw)
        pages = [sanitize_utf8(page.extract_text()) for page in pdf_reader.pages]
    except Exception as error:  # noqa: BLE001
        error_type = type(error).__name__
        logger.warning(
            "A pdf reader error occurred in %s: %s - %s",
            raw_name,
            error_type,
            error,
        )
    return pages


@udf(
    params=(Object(load_pdf_pages), *tuple(DatasetRow.schema.keys())),
    output={**DatasetRow.schema, "n_page": Int, "total_pages": Int, "page": String},
)
def pdf_pages(pages, *args):
    record = dict(zip(DatasetRow.schema.keys(), args))
    del record["random"]  # random will be populated automatically
    record["is_latest"] = record["is_latest"] > 0  # needs to be a bool

    total_pages = len(pages)
    for n_page, page in enumerate(pages):
        page = sanitize_ascii(page)
        page = compress_whitespace(page)
        yield (*DatasetRow.create(**record), n_page, total_pages, page)


def load_bibtex(raw):
    bibtex_str = raw.read().decode("utf-8")
    parser = bibtex.Parser()
    return parser.parse_string(bibtex_str)
