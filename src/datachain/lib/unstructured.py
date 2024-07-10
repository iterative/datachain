import shutil
import tempfile

from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dataframe

from datachain.lib.udf import Mapper
from datachain.query import Stream
from datachain.sql.types import JSON, String


class PartitionObject(Mapper):
    def __init__(self):
        super().__init__(
            [
                Stream(),
            ],
            {
                "elements": JSON,
                "title": String,
                "text": String,
                "error": String,
            },
        )

    def encode_object(self, raw):
        fname = str(raw).replace(">", "").replace("<", "")
        output = tempfile.TemporaryFile()
        shutil.copyfileobj(raw, output)
        elements = partition(file=output, metadata_filename=fname)
        output.close()
        return elements

    def __call__(self, stream):
        with stream:
            elements = self.encode_object(stream)

        title = str(elements[0])
        text = "\n\n".join([str(el) for el in elements])
        df = convert_to_dataframe(elements)
        return (df.to_json(), title, text, "")
