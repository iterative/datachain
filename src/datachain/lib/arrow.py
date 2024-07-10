import re
from typing import TYPE_CHECKING, Optional

from pyarrow.dataset import dataset

from datachain.lib.file import File, IndexedFile
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    import pyarrow as pa


class ArrowGenerator(Generator):
    def __init__(self, schema: Optional["pa.Schema"] = None, **kwargs):
        """
        Generator for getting rows from tabular files.

        Parameters:

        schema : Optional pyarrow schema for validation.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.schema = schema
        self.kwargs = kwargs

    def process(self, file: File):
        path = file.get_path()
        ds = dataset(path, filesystem=file.get_fs(), schema=self.schema, **self.kwargs)
        index = 0
        for record_batch in ds.to_batches():
            for record in record_batch.to_pylist():
                source = IndexedFile(file=file, index=index)
                yield [source, *record.values()]
                index += 1


def schema_to_output(schema: "pa.Schema"):
    """Generate UDF output schema from pyarrow schema."""
    default_column = 0
    output = {"source": IndexedFile}
    for field in schema:
        column = field.name.lower()
        column = re.sub("[^0-9a-z_]+", "", column)
        if not column:
            column = f"c{default_column}"
            default_column += 1
        output[column] = _arrow_type_mapper(field.type)  # type: ignore[assignment]

    return output


def _arrow_type_mapper(col_type: "pa.DataType") -> type:  # noqa: PLR0911
    """Convert pyarrow types to basic types."""
    from datetime import datetime

    import pyarrow as pa

    if pa.types.is_timestamp(col_type):
        return datetime
    if pa.types.is_binary(col_type):
        return bytes
    if pa.types.is_floating(col_type):
        return float
    if pa.types.is_integer(col_type):
        return int
    if pa.types.is_boolean(col_type):
        return bool
    if pa.types.is_date(col_type):
        return datetime
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return str
    if pa.types.is_list(col_type):
        return list[_arrow_type_mapper(col_type.value_type)]  # type: ignore[misc]
    if pa.types.is_struct(col_type) or pa.types.is_map(col_type):
        return dict
    if isinstance(col_type, pa.lib.DictionaryType):
        return _arrow_type_mapper(col_type.value_type)  # type: ignore[return-value]
    raise TypeError(f"{col_type!r} datatypes not supported")
