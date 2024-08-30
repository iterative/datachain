from datetime import datetime
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from datachain.lib.arrow import (
    ArrowGenerator,
    arrow_type_mapper,
    schema_to_output,
)
from datachain.lib.data_model import dict_to_data_model
from datachain.lib.file import File, IndexedFile


def test_arrow_generator(tmp_path, catalog):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    df = pd.DataFrame({"id": ids, "text": texts})

    name = "111.parquet"
    pq_path = tmp_path / name
    df.to_parquet(pq_path)
    stream = File(path=pq_path.as_posix(), source="file:///")
    stream._set_stream(catalog, caching_enabled=False)

    func = ArrowGenerator()
    objs = list(func.process(stream))

    assert len(objs) == len(ids)
    for index, (o, id, text) in enumerate(zip(objs, ids, texts)):
        assert isinstance(o[0], IndexedFile)
        assert isinstance(o[0].file, File)
        assert o[0].index == index
        assert o[1] == id
        assert o[2] == text


def test_arrow_generator_no_source(tmp_path, catalog):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    df = pd.DataFrame({"id": ids, "text": texts})

    name = "111.parquet"
    pq_path = tmp_path / name
    df.to_parquet(pq_path)
    stream = File(path=pq_path.as_posix(), source="file:///")
    stream._set_stream(catalog, caching_enabled=False)

    func = ArrowGenerator(source=False)
    objs = list(func.process(stream))

    for o, id, text in zip(objs, ids, texts):
        assert o[0] == id
        assert o[1] == text


def test_arrow_generator_output_schema(tmp_path, catalog):
    ids = [12345, 67890, 34, 0xF0123]
    texts = ["28", "22", "we", "hello world"]
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}, {"a": 7, "b": 8}]
    df = pd.DataFrame({"id": ids, "text": texts, "dict": dicts})
    table = pa.Table.from_pandas(df)

    name = "111.parquet"
    pq_path = tmp_path / name
    pq.write_table(table, pq_path)
    stream = File(path=pq_path.as_posix(), source="file:///")
    stream._set_stream(catalog, caching_enabled=False)

    output_schema = dict_to_data_model("", schema_to_output(table.schema))
    func = ArrowGenerator(output_schema=output_schema)
    objs = list(func.process(stream))

    assert len(objs) == len(ids)
    for index, (o, id, text, dict) in enumerate(zip(objs, ids, texts, dicts)):
        assert isinstance(o[0], IndexedFile)
        assert isinstance(o[0].file, File)
        assert o[0].index == index
        assert o[1].id == id
        assert o[1].text == text
        assert o[1].dict.a == dict["a"]
        assert o[1].dict.b == dict["b"]


@pytest.mark.parametrize(
    "col_type,expected",
    (
        (pa.timestamp("us"), datetime),
        (pa.binary(), bytes),
        (pa.float32(), float),
        (pa.float64(), float),
        (pa.float16(), float),
        (pa.int8(), int),
        (pa.int16(), int),
        (pa.int32(), int),
        (pa.int64(), int),
        (pa.uint8(), int),
        (pa.bool_(), bool),
        (pa.date32(), datetime),
        (pa.string(), str),
        (pa.large_string(), str),
        (pa.map_(pa.string(), pa.int32()), dict),
        (pa.dictionary(pa.int64(), pa.string()), str),
        (pa.list_(pa.string()), list[str]),
    ),
)
def test_arrow_type_mapper(col_type, expected):
    assert arrow_type_mapper(col_type) == expected


def test_arrow_type_mapper_struct():
    col_type = pa.struct({"x": pa.int32(), "y": pa.string()})
    fields = arrow_type_mapper(col_type).model_fields
    assert list(fields.keys()) == ["x", "y"]
    dtypes = [field.annotation for field in fields.values()]
    assert dtypes == [Optional[int], Optional[str]]


def test_arrow_type_error():
    col_type = pa.union(
        [pa.field("a", pa.binary(10)), pa.field("b", pa.string())],
        mode=pa.lib.UnionMode_DENSE,
    )
    with pytest.raises(TypeError):
        arrow_type_mapper(col_type)


def test_schema_to_output():
    schema = pa.schema(
        [
            ("some_int", pa.int32()),
            ("some_string", pa.string()),
            ("strict_int", pa.int32(), False),
        ]
    )
    assert schema_to_output(schema) == {
        "some_int": Optional[int],
        "some_string": Optional[str],
        "strict_int": int,
    }


def test_parquet_convert_column_names():
    schema = pa.schema(
        [
            ("UpperCaseCol", pa.int32()),
            ("dot.notation.col", pa.int32()),
            ("with-dashes", pa.int32()),
            ("with spaces", pa.int32()),
        ]
    )
    assert list(schema_to_output(schema)) == [
        "uppercasecol",
        "dotnotationcol",
        "withdashes",
        "withspaces",
    ]


def test_parquet_missing_column_names():
    schema = pa.schema(
        [
            ("", pa.int32()),
            ("", pa.int32()),
        ]
    )
    assert list(schema_to_output(schema)) == ["c0", "c1"]


def test_parquet_override_column_names():
    schema = pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])
    col_names = ["n1", "n2"]
    assert schema_to_output(schema, col_names) == {
        "n1": Optional[int],
        "n2": Optional[str],
    }


def test_parquet_override_column_names_invalid():
    schema = pa.schema([("some_int", pa.int32()), ("some_string", pa.string())])
    col_names = ["n1", "n2", "n3"]
    with pytest.raises(ValueError):
        schema_to_output(schema, col_names)
