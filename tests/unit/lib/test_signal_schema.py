import json
from datetime import datetime
from typing import Any, Dict, Final, List, Literal, Optional, Union  # noqa: UP035

import pytest

from datachain import Column, DataModel
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import File, TextFile
from datachain.lib.signal_schema import (
    SetupError,
    SignalResolvingError,
    SignalSchema,
    SignalSchemaError,
    SignalSchemaWarning,
)
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Float32,
    Float64,
    Int,
    Int32,
    Int64,
    String,
    UInt32,
    UInt64,
)


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


class MyTypeComplex(DataModel):
    name: str
    items: list[MyType1]
    lookup: dict[str, MyType2]


class MyTypeComplexOld(DataModel):
    name: str
    items: List[MyType1]  # noqa: UP006
    lookup: Dict[str, MyType2]  # noqa: UP006


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
        SignalSchema.deserialize({"name": "Union[str,"})

    with pytest.warns(SignalSchemaWarning):
        # Warn if unknown fields are encountered - don't throw an exception to ensure
        # that all data can be shown.
        SignalSchema.deserialize({"name": "unknown"})


def test_serialize_simple():
    schema = {
        "name": str,
        "age": float,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 2
    assert signals["name"] == "str"
    assert signals["age"] == "float"
    assert "_custom_types" not in signals


def test_serialize_basic():
    schema = {
        "name": str,
        "age": float,
        "f": File,
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 4
    assert signals["name"] == "str"
    assert signals["age"] == "float"
    assert signals["f"] == "File@v1"
    assert "File@v1" in signals["_custom_types"]


def test_feature_schema_serialize_optional():
    schema = {
        "name": Optional[str],
        "feature": Optional[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["feature"] == "Union[MyType1, NoneType]"
    assert signals["_custom_types"] == {"MyType1@v1": {"aa": "int", "bb": "str"}}


def test_feature_schema_serialize_list():
    schema = {
        "name": Optional[str],
        "features": list[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["features"] == "list[MyType1]"
    assert signals["_custom_types"] == {"MyType1@v1": {"aa": "int", "bb": "str"}}


def test_feature_schema_serialize_list_old():
    schema = {
        "name": Optional[str],
        "features": List[MyType1],  # noqa: UP006
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["features"] == "list[MyType1]"
    assert signals["_custom_types"] == {"MyType1@v1": {"aa": "int", "bb": "str"}}


def test_feature_schema_serialize_nested_types():
    schema = {
        "name": Optional[str],
        "feature_nested": Optional[MyType2],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["feature_nested"] == "Union[MyType2, NoneType]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {"aa": "int", "bb": "str"},
        "MyType2@v1": {"deep": "MyType1@v1", "name": "str"},
    }


def test_feature_schema_serialize_nested_duplicate_types():
    schema = {
        "name": Optional[str],
        "feature_nested": Optional[MyType2],
        "feature_not_nested": Optional[MyType1],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 4
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["feature_nested"] == "Union[MyType2, NoneType]"
    assert signals["feature_not_nested"] == "Union[MyType1, NoneType]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {"aa": "int", "bb": "str"},
        "MyType2@v1": {"deep": "MyType1@v1", "name": "str"},
    }


def test_feature_schema_serialize_complex():
    schema = {
        "name": Optional[str],
        "feature": Optional[MyTypeComplex],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["feature"] == "Union[MyTypeComplex, NoneType]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {"aa": "int", "bb": "str"},
        "MyType2@v1": {"deep": "MyType1@v1", "name": "str"},
        "MyTypeComplex@v1": {
            "name": "str",
            "items": "list[MyType1]",
            "lookup": "dict[str, MyType2]",
        },
    }


def test_feature_schema_serialize_complex_old():
    schema = {
        "name": Optional[str],
        "feature": Optional[MyTypeComplexOld],
    }
    signals = SignalSchema(schema).serialize()

    assert len(signals) == 3
    assert signals["name"] == "Union[str, NoneType]"
    assert signals["feature"] == "Union[MyTypeComplexOld, NoneType]"
    assert signals["_custom_types"] == {
        "MyType1@v1": {"aa": "int", "bb": "str"},
        "MyType2@v1": {"deep": "MyType1@v1", "name": "str"},
        "MyTypeComplexOld@v1": {
            "name": "str",
            "items": "list[MyType1]",
            "lookup": "dict[str, MyType2]",
        },
    }


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

    assert "f__path" in spec
    assert spec["f__path"] == String

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


def test_select_custom_type():
    schema = SignalSchema.deserialize(
        {
            "age": "float",
            "address": "str",
            "f": "ExternalCustomType1@v1",
            "_custom_types": {"ExternalCustomType1@v1": {"aa": "int", "bb": "str"}},
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


def test_select_nested_names_custom_types():
    schema = SignalSchema.deserialize(
        {
            "address": "str",
            "fr": "NestedType2@v1",
            "_custom_types": {
                "NestedType1@v1": {"aa": "int", "bb": "str"},
                "NestedType2@v1": {"deep": "NestedType1@v1", "name": "str"},
            },
        }
    )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    # This is a dynamically restored model
    nested_type_1 = fr_signals["fr.deep"]
    assert issubclass(nested_type_1, DataModel)
    assert {n: fi.annotation for n, fi in nested_type_1.model_fields.items()} == {
        "aa": int,
        "bb": str,
    }

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


def test_select_complex_names_custom_types():
    with pytest.warns(SignalSchemaWarning):
        schema = SignalSchema.deserialize(
            {
                "address": "str",
                "fr": "ComplexType@v1",
                "_custom_types": {
                    "NestedTypeComplex@v1": {
                        "aa": "float",
                        "bb": "bytes",
                        "items": "list[Union[dict[str, float], dict[str, int]]]",
                        "maybe_texts": "Union[list[Any], dict[str, Any], NoneType]",
                        "anything": "UnknownCustomType",
                    },
                    "ComplexType@v1": {"deep": "NestedTypeComplex@v1", "name": "str"},
                },
            }
        )

    fr_signals = schema.resolve("fr.deep").values
    assert "fr.deep" in fr_signals
    # This is a dynamically restored model
    nested_type_complex = fr_signals["fr.deep"]
    assert issubclass(nested_type_complex, DataModel)
    assert {n: fi.annotation for n, fi in nested_type_complex.model_fields.items()} == {
        "aa": float,
        "bb": bytes,
        "items": list[Union[dict[str, float], dict[str, int]]],
        "maybe_texts": Union[list[Any], dict[str, Any], None],
        "anything": Any,
    }

    basic_signals = schema.resolve(
        "fr.deep.aa", "fr.deep.bb", "fr.deep.maybe_texts", "fr.deep.anything"
    ).values
    assert "fr.deep.aa" in basic_signals
    assert "fr.deep.bb" in basic_signals
    assert "fr.deep.maybe_texts" in basic_signals
    assert "fr.deep.anything" in basic_signals
    assert basic_signals["fr.deep.aa"] is float
    assert basic_signals["fr.deep.bb"] is bytes
    assert (
        basic_signals["fr.deep.maybe_texts"] is Union[list[Any], dict[str, Any], None]
    )
    assert basic_signals["fr.deep.anything"] is Any


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


def test_get_features_nested(test_session, nested_file_schema):
    file = File(path="test")
    file_dict = file.model_dump()
    file_vals = list(file_dict.values())
    nested_file = nested_file_schema.values["my_f"](
        ref="str", nested_file=file, **file_dict
    )
    expected_features = ["str", 0.0, file, nested_file]
    row = ["str", 0.0, *file_vals, *file_vals, "str", *file_vals]
    actual_features = nested_file_schema.row_to_features(row, test_session.catalog)
    assert expected_features == actual_features
    assert actual_features[2]._catalog == test_session.catalog
    assert actual_features[3]._catalog == test_session.catalog
    assert actual_features[3].nested_file._catalog == test_session.catalog


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
        None: "NoneType",
        MyType2: "MyType2",
        Any: "Any",
        Literal: "Literal",
        Final: "Final",
        Optional[MyType2]: "Union[MyType2, NoneType]",
        Union[str, int]: "Union[str, int]",
        Union[str, int, bool]: "Union[str, int, bool]",
        Union[Optional[MyType2]]: "Union[MyType2, NoneType]",
        list: "list",
        list[bool]: "list[bool]",
        List[bool]: "list[bool]",  # noqa: UP006
        list[Optional[bool]]: "list[Union[bool, NoneType]]",
        List[Optional[bool]]: "list[Union[bool, NoneType]]",  # noqa: UP006
        dict: "dict",
        dict[str, bool]: "dict[str, bool]",
        Dict[str, bool]: "dict[str, bool]",  # noqa: UP006
        dict[str, Optional[MyType1]]: "dict[str, Union[MyType1, NoneType]]",
        Dict[str, Optional[MyType1]]: "dict[str, Union[MyType1, NoneType]]",  # noqa: UP006
        Union[str, list[str]]: "Union[str, list[str]]",
        Union[str, List[str]]: "Union[str, list[str]]",  # noqa: UP006
        Optional[Literal["x"]]: "Union[Literal, NoneType]",
        Optional[list[bytes]]: "Union[list[bytes], NoneType]",
        Optional[List[bytes]]: "Union[list[bytes], NoneType]",  # noqa: UP006
        list[Any]: "list[Any]",
        List[Any]: "list[Any]",  # noqa: UP006
    }

    for t, v in mapping.items():
        assert SignalSchema._type_to_str(t) == v

    # Test that unknown types are ignored, but raise a warning.
    mapping_warnings = {
        5: "Any",
        "UnknownType": "Any",
    }
    for t, v in mapping_warnings.items():
        with pytest.warns(SignalSchemaWarning):
            assert SignalSchema._type_to_str(t) == v


def test_resolve_types():
    mapping = {
        "int": int,
        "float": float,
        "NoneType": None,
        "MyType2@v1": MyType2,
        "Any": Any,
        "Literal": Any,
        "Final": Final,
        "Union[MyType2@v1, NoneType]": Optional[MyType2],
        "Optional[MyType2@v1]": Optional[MyType2],
        "Union[str, int]": Union[str, int],
        "Union[str, int, bool]": Union[str, int, bool],
        "Union[Optional[MyType2@v1]]": Union[Optional[MyType2]],
        "list": list,
        "list[bool]": list[bool],
        "List[bool]": list[bool],
        "list[Union[bool, NoneType]]": list[Optional[bool]],
        "List[Union[bool, NoneType]]": list[Optional[bool]],
        "list[Optional[bool]]": list[Optional[bool]],
        "List[Optional[bool]]": list[Optional[bool]],
        "dict": dict,
        "dict[str, bool]": dict[str, bool],
        "Dict[str, bool]": dict[str, bool],
        "dict[str, Union[MyType1@v1, NoneType]]": dict[str, Optional[MyType1]],
        "Dict[str, Union[MyType1@v1, NoneType]]": dict[str, Optional[MyType1]],
        "dict[str, Optional[MyType1@v1]]": dict[str, Optional[MyType1]],
        "Dict[str, Optional[MyType1@v1]]": dict[str, Optional[MyType1]],
        "Union[str, list[str]]": Union[str, list[str]],
        "Union[str, List[str]]": Union[str, list[str]],
        "Union[Literal, NoneType]": Optional[Any],
        "Union[list[bytes], NoneType]": Optional[list[bytes]],
        "Union[List[bytes], NoneType]": Optional[list[bytes]],
    }

    for s, t in mapping.items():
        assert SignalSchema._resolve_type(s, {}) == t

    # Test that unknown types are ignored, but raise a warning.
    mapping_warnings = {
        "BogusType": Any,
        "UnknownType": Any,
        "list[UnknownType]": list[Any],
        "List[UnknownType]": list[Any],
    }
    for s, t in mapping_warnings.items():
        with pytest.warns(SignalSchemaWarning):
            assert SignalSchema._resolve_type(s, {}) == t


def test_resolve_types_errors():
    bogus_types_messages = {
        "": r"cannot be empty",
        "[str]": r"cannot start with '\['",
        "Union[str": r"Unclosed square bracket",
        "Union]str[": r"Square brackets are out of order",
        "Union[]": r"Empty square brackets",
        "Union[str, int]]": r"Extra closing square bracket",
        "Union[str, Optional[int]": r"Unclosed square bracket",
    }

    for t, m in bogus_types_messages.items():
        with pytest.raises(TypeError, match=m):
            SignalSchema._resolve_type(t, {})


def test_db_signals():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).db_signals())

    assert lst == [
        "name",
        "age",
        "fr__name",
        "fr__deep__aa",
        "fr__deep__bb",
    ]


def test_db_signals_filtering_by_name():
    schema = SignalSchema({"name": str, "age": float, "fr": MyType2})

    assert list(schema.db_signals(name="fr")) == [
        "fr__name",
        "fr__deep__aa",
        "fr__deep__bb",
    ]
    assert list(schema.db_signals(name="name")) == ["name"]
    assert list(schema.db_signals(name="missing")) == []


def test_db_signals_as_columns():
    spec = {"name": str, "age": float, "fr": MyType2}
    lst = list(SignalSchema(spec).db_signals(as_columns=True))

    assert all(isinstance(s, Column) for s in lst)

    assert [(c.name, type(c.type)) for c in lst] == [
        ("name", String),
        ("age", Float),
        ("fr__name", String),
        ("fr__deep__aa", Int64),
        ("fr__deep__bb", String),
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


def test_mutate_rename():
    schema = SignalSchema({"name": str})
    schema = schema.mutate({"new_name": Column("name")})
    assert schema.values == {"new_name": str}


def test_mutate_rename_leaf(nested_file_schema):
    schema = nested_file_schema.mutate({"new_name": Column("my_f__nested_file")})
    assert schema.values == {**nested_file_schema.values, "new_name": File}


def test_mutate_new_signal():
    schema = SignalSchema({"name": str})
    schema = schema.mutate({"age": Column("age", Float)})
    assert schema.values == {"name": str, "age": float}


def test_mutate_change_type():
    schema = SignalSchema({"name": str, "age": float, "f": File})
    schema = schema.mutate({"age": int, "f": TextFile})
    assert schema.values == {"name": str, "age": int, "f": TextFile}


@pytest.mark.parametrize(
    "column_type,signal_type",
    [
        [String, str],
        [Boolean, bool],
        [Int, int],
        [Int32, int],
        [UInt32, int],
        [Int64, int],
        [UInt64, int],
        [Float, float],
        [Float32, float],
        [Float64, float],
        [Array(Int), list],
        [JSON, dict],
        [DateTime, datetime],
        [Binary, bytes],
    ],
)
def test_column_types(column_type, signal_type):
    signals = SignalSchema.from_column_types({"val": column_type}).values

    assert len(signals) == 1
    assert signals["val"] is signal_type
