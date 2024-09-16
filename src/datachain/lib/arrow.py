import re
from collections.abc import Sequence
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Optional

import pyarrow as pa
from pyarrow.dataset import dataset
from tqdm import tqdm

from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import File, IndexedFile
from datachain.lib.model_store import ModelStore
from datachain.lib.udf import Generator

if TYPE_CHECKING:
    from datasets.features.features import Features
    from pydantic import BaseModel

    from datachain.lib.data_model import DataType
    from datachain.lib.dc import DataChain


class ArrowGenerator(Generator):
    def __init__(
        self,
        input_schema: Optional["pa.Schema"] = None,
        output_schema: Optional[type["BaseModel"]] = None,
        source: bool = True,
        nrows: Optional[int] = None,
        **kwargs,
    ):
        """
        Generator for getting rows from tabular files.

        Parameters:

        input_schema : Optional pyarrow schema for validation.
        output_schema : Optional pydantic model for validation.
        source : Whether to include info about the source file.
        nrows : Optional row limit.
        kwargs: Parameters to pass to pyarrow.dataset.dataset.
        """
        super().__init__()
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.source = source
        self.nrows = nrows
        self.kwargs = kwargs

    def process(self, file: File):
        if file._caching_enabled:
            path = file.get_local_path(download=True)
            ds = dataset(path, schema=self.input_schema, **self.kwargs)
        elif self.nrows:
            path = _nrows_file(file, self.nrows)
            ds = dataset(path, schema=self.input_schema, **self.kwargs)
        else:
            path = file.get_path()
            ds = dataset(
                path, filesystem=file.get_fs(), schema=self.input_schema, **self.kwargs
            )
        hf_schema = _get_hf_schema(ds.schema)
        index = 0
        with tqdm(desc="Parsed by pyarrow", unit=" rows") as pbar:
            for record_batch in ds.to_batches():
                for record in record_batch.to_pylist():
                    vals = list(record.values())
                    if self.output_schema:
                        fields = self.output_schema.model_fields
                        vals_dict = {}
                        for i, ((field, field_info), val) in enumerate(
                            zip(fields.items(), vals)
                        ):
                            anno = field_info.annotation
                            if hf_schema:
                                from datachain.lib.hf import convert_feature

                                feat = list(hf_schema[0].values())[i]
                                vals_dict[field] = convert_feature(val, feat, anno)
                            elif ModelStore.is_pydantic(anno):
                                vals_dict[field] = anno(**val)  # type: ignore[misc]
                            else:
                                vals_dict[field] = val
                        vals = [self.output_schema(**vals_dict)]
                    if self.source:
                        yield [IndexedFile(file=file, index=index), *vals]
                    else:
                        yield vals
                    index += 1
                pbar.update(len(record_batch))


def infer_schema(chain: "DataChain", **kwargs) -> pa.Schema:
    schemas = []
    for file in chain.collect("file"):
        ds = dataset(file.get_path(), filesystem=file.get_fs(), **kwargs)  # type: ignore[union-attr]
        schemas.append(ds.schema)
    return pa.unify_schemas(schemas)


def schema_to_output(schema: pa.Schema, col_names: Optional[Sequence[str]] = None):
    """Generate UDF output schema from pyarrow schema."""
    if col_names and (len(schema) != len(col_names)):
        raise ValueError(
            "Error generating output from Arrow schema - "
            f"Schema has {len(schema)} columns but got {len(col_names)} column names."
        )
    if not col_names:
        col_names = schema.names
    columns = _convert_col_names(col_names)  # type: ignore[arg-type]
    hf_schema = _get_hf_schema(schema)
    if hf_schema:
        return {
            column: hf_type for hf_type, column in zip(hf_schema[1].values(), columns)
        }
    output = {}
    for field, column in zip(schema, columns):
        dtype = arrow_type_mapper(field.type, column)  # type: ignore[assignment]
        if field.nullable and not ModelStore.is_pydantic(dtype):
            dtype = Optional[dtype]  # type: ignore[assignment]
        output[column] = dtype
    return output


def _convert_col_names(col_names: Sequence[str]) -> list[str]:
    default_column = 0
    converted_col_names = []
    for column in col_names:
        column = column.lower()
        column = re.sub("[^0-9a-z_]+", "", column)
        if not column:
            column = f"c{default_column}"
            default_column += 1
        converted_col_names.append(column)
    return converted_col_names


def arrow_type_mapper(col_type: pa.DataType, column: str = "") -> type:  # noqa: PLR0911
    """Convert pyarrow types to basic types."""
    from datetime import datetime

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
        return list[arrow_type_mapper(col_type.value_type)]  # type: ignore[return-value, misc]
    if pa.types.is_struct(col_type):
        type_dict = {}
        for field in col_type:
            dtype = arrow_type_mapper(field.type, field.name)
            if field.nullable and not ModelStore.is_pydantic(dtype):
                dtype = Optional[dtype]  # type: ignore[assignment]
            type_dict[field.name] = dtype
        return dict_to_data_model(column, type_dict)
    if pa.types.is_map(col_type):
        return dict
    if isinstance(col_type, pa.lib.DictionaryType):
        return arrow_type_mapper(col_type.value_type)  # type: ignore[return-value]
    raise TypeError(f"{col_type!r} datatypes not supported")


def _nrows_file(file: File, nrows: int) -> str:
    tf = NamedTemporaryFile(delete=False)  # noqa: SIM115
    with file.open(mode="r") as reader:
        with open(tf.name, "a") as writer:
            for row, line in enumerate(reader):
                if row >= nrows:
                    break
                writer.write(line)
                writer.write("\n")
    return tf.name


def _get_hf_schema(
    schema: "pa.Schema",
) -> Optional[tuple["Features", dict[str, "DataType"]]]:
    if schema.metadata and b"huggingface" in schema.metadata:
        from datachain.lib.hf import get_output_schema, schema_from_arrow

        features = schema_from_arrow(schema)
        return features, get_output_schema(features)
    return None
