import json
from typing import Optional, Union

import pytest

from datachain import DataModel
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File
from datachain.lib.signal_schema import (
    SetupError,
    SignalResolvingError,
    SignalSchema,
    SignalSchemaError,
)
from datachain.sql.types import Float, Int64, String


@pytest.fixture
def nested_file_schema():
    class _MyFile(File):
        ref: str
        nested_file: File

    schema = {"name": str, "age": float, "f": File, "my_f": _MyFile}

    return SignalSchema(schema)


class MyType1(DataModel):
    aa: int
    bb: str


class MyType2(DataModel):
    name: str
    deep: MyType1


def test_deserialize_basic():
    stored = {"name": "str", "count": "int", "file": "File@v1"}
    signals = SignalSchema.deserialize(stored)

    assert len(signals.values) == 3
    assert signals.values.keys() == stored.keys()
    assert list(signals.values.values()) == [str, int, File]


def test_deserialize_error():
    SignalSchema.deserialize({})

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize(json.dumps({"name": "str"}))

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize({"name": [1, 2, 3]})

    with pytest.raises(SignalSchemaError):
        SignalSchema.deserialize({"name": "unknown"})


def test_serialize_basic():
    schema = {
        "name": str,
        "age": float,
        "f": File,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "str"
    assert signals["age"] == "float"
    assert signals["f"] == "File@v1"


def test_feature_schema_serialize_optional():
    schema = {
        "name": Optional[str],
        "feature": Optional[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 2
    assert signals["name"] == "str"
    assert signals["feature"] == "MyType1"


def test_serialize_from_column():
    signals = SignalSchema.from_column_types({"age": Float, "name": String}).values

    assert len(signals) == 2
    assert signals["name"] is str
    assert signals["age"] is float


def test_serialize_from_column_error():
    with pytest.raises(SignalSchemaError):
        SignalSchema.from_column_types({"age": Float, "wrong_type": File})


def test_to_udf_spec():
    signals = SignalSchema.deserialize(
        {
            "age": "float",
            "address": "str",
            "f": "File@v1",
        }
    )

    spec = SignalSchema.to_udf_spec(signals)

    assert len(spec) == 2 + len(File.model_fields)

    assert "age" in spec
    assert spec["age"] == Float

    assert "address" in spec
    assert spec["address"] == String

    assert "f__name" in spec
    assert spec["f__name"] == String

    assert "f__size" in spec
    assert spec["f__size"] == Int64


def test_select():
    schema = SignalSchema.deserialize(
        {
            "age": "float",
            "address": "str",
            "f": "MyType1@v1",
        }
    )

    new = schema.resolve("age", "f.aa", "f.bb")
    assert isinstance(new, SignalSchema)

    signals = new.values
    assert len(signals) == 3
    assert {"age", "f.aa", "f.bb"} == signals.keys()
    assert signals["age"] is float
    assert signals["f.aa"] is int
    assert signals["f.bb"] is str


def test_select_nested_names():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "MyType2@v1",
        }
    )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    assert fr_signals["fr.deep"] == MyType1

    basic_signals = schema.resolve("fr.deep.aa", "fr.deep.bb").values
    assert "fr.deep.aa" in basic_signals
    assert "fr.deep.bb" in basic_signals
    assert basic_signals["fr.deep.aa"] is int
    assert basic_signals["fr.deep.bb"] is str


def test_select_nested_errors():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "MyType2@v1",
        }
    )

    schema = schema.resolve("fr.deep.aa", "fr.deep.bb")

    with pytest.raises(SignalResolvingError):
        schema.resolve("some_random")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr.deep")

    with pytest.raises(SignalResolvingError):
        schema.resolve("fr.deep.not_exist")


def test_get_signals_basic():
    schema = {
        "name": str,
        "age": float,
        "f": File,
    }
    assert list(SignalSchema(schema).get_signals(File)) == ["f"]


def test_get_signals_no_signal():
    schema = {
        "name": str,
    }
    assert list(SignalSchema(schema).get_signals(File)) == []


def test_get_signals_nested(nested_file_schema):
    files = list(nested_file_schema.get_signals(File))
    assert files == ["f", "my_f", "my_f.nested_file"]


def test_get_signals_subclass(nested_file_schema):
    class NewFile(File):
        pass

    schema = {
        "name": str,
        "age": float,
        "f": NewFile,
    }
    assert list(SignalSchema(schema).get_signals(File)) == ["f"]


def test_build_tree():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).get_flat_tree())

    assert lst == [
        (["name"], str, False, 0),
        (["age"], float, False, 0),
        (["fr"], MyType2, True, 0),
        (["fr", "name"], str, False, 1),
        (["fr", "deep"], MyType1, True, 1),
        (["fr", "deep", "aa"], int, False, 2),
        (["fr", "deep", "bb"], str, False, 2),
    ]


def test_print_types():
    mapping = {
        int: "int",
        float: "float",
        MyType2: "MyType2",
        Optional[MyType2]: "Union[MyType2, NoneType]",
        Union[str, int]: "Union[str, int]",
        Union[Optional[MyType2]]: "Union[MyType2, NoneType]",
        list: "list",
        list[Optional[bool]]: "list[Union[bool, NoneType]]",
        dict: "dict",
        dict[str, Optional[MyType1]]: "dict[str, Union[MyType1, NoneType]]",
    }

    for t, v in mapping.items():
        assert SignalSchema._type_to_str(t) == v


def test_bd_signals():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).db_signals())

    assert lst == [
        "name",
        "age",
        "fr__name",
        "fr__deep__aa",
        "fr__deep__bb",
    ]


def test_row_to_objs():
    spec = {"name": str, "age": float, "fr": MyType2}
    schema = SignalSchema(spec)

    val = MyType2(name="Fred", deep=MyType1(aa=129, bb="qwe"))
    row = ("myname", 12.5, *flatten(val))

    res = schema.row_to_objs(row)

    assert res == ["myname", 12.5, val]


def test_row_to_objs_setup():
    spec = {"name": str, "age": float, "init_val": int, "fr": MyType2}
    setup_value = 84635
    setup = {"init_val": lambda: setup_value}
    schema = SignalSchema(spec, setup)

    val = MyType2(name="Fred", deep=MyType1(aa=129, bb="qwe"))
    row = ("myname", 12.5, *flatten(val))

    res = schema.row_to_objs(row)
    assert res == ["myname", 12.5, setup_value, val]


def test_setup_not_callable():
    spec = {"name": str, "age": float, "init_val": int, "fr": MyType2}
    setup_dict = {"init_val": "asdfd"}
    with pytest.raises(SetupError):
        SignalSchema(spec, setup_dict)


def test_slice():
    schema = {"name": str, "age": float, "address": str}
    keys = ["age", "name"]
    sliced = SignalSchema(schema).slice(keys)
    assert list(sliced.values.items()) == [("age", float), ("name", str)]


def test_slice_nested():
    schema = {
        "name": str,
        "feature": MyType1,
    }
    keys = ["feature.aa"]
    sliced = SignalSchema(schema).slice(keys)
    assert list(sliced.values.items()) == [("feature.aa", int)]
