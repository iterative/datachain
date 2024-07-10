from typing import ClassVar, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from datachain.lib.feature import Feature
from datachain.lib.feature_registry import Registry
from datachain.lib.feature_utils import pydantic_to_feature
from datachain.lib.signal_schema import SignalSchema
from datachain.sql.types import (
    Array,
    Int64,
    String,
)


class FileBasic(Feature):
    parent: str = Field(default="")
    name: str
    size: int = Field(default=0)


class TestFileInfo(FileBasic):
    location: dict = Field(default={})


class FileInfoEx(Feature):
    f_info: TestFileInfo
    type_id: int


class MyNestedClass(Feature):
    type: int
    Name: str = Field(default="test1")


class MyTest(Feature):
    ThisIsName: str
    subClass: MyNestedClass  # noqa: N815


def test_flatten_basic():
    vals = FileBasic(parent="hello", name="world", size=123)._flatten()
    assert vals == ("hello", "world", 123)


def test_flatten_with_json():
    t1 = TestFileInfo(parent="prt4", name="test1", size=42, location={"ee": "rr"})
    assert t1._flatten() == ("prt4", "test1", 42, {"ee": "rr"})


def test_flatten_with_empty_json():
    with pytest.raises(ValidationError):
        TestFileInfo(parent="prt4", name="test1", size=42, location=None)


def test_flatten_with_accepted_empty_json():
    class _Test(Feature):
        d: Optional[dict]

    assert _Test(d=None)._flatten() == (None,)


def test_flatten_nested():
    t0 = TestFileInfo(parent="sfo", name="sf", size=567, location={"42": 999})
    t1 = FileInfoEx(f_info=t0, type_id=1849)

    assert t1._flatten() == ("sfo", "sf", 567, {"42": 999}, 1849)


def test_flatten_list():
    t1 = TestFileInfo(parent="p1", name="n4", size=3, location={"a": "b"})
    t2 = TestFileInfo(parent="p2", name="n5", size=2, location={"c": "d"})

    vals = t1._flatten_list([t1, t2])
    assert vals == ("p1", "n4", 3, {"a": "b"}, "p2", "n5", 2, {"c": "d"})


def test_registry():
    class MyTestRndmz(Feature):
        name: str
        count: int

    assert Registry.get(MyTestRndmz.__name__) == MyTestRndmz
    assert Registry.get(MyTestRndmz.__name__, version=1) == MyTestRndmz
    Registry.remove(MyTestRndmz)


def test_registry_versioned():
    class MyTestXYZ(Feature):
        _version: ClassVar[int] = 42
        name: str
        count: int

    assert Registry.get(MyTestXYZ.__name__) == MyTestXYZ
    assert Registry.get(MyTestXYZ.__name__, version=1) is None
    assert Registry.get(MyTestXYZ.__name__, version=42) == MyTestXYZ
    Registry.remove(MyTestXYZ)


def test_inheritance():
    class SubObject(Feature):
        subname: str

    class SoMyTest1(Feature):
        name: str
        sub: SubObject

    class SoMyTest2(SoMyTest1):
        pass

    try:
        with pytest.raises(ValueError):
            SoMyTest2()

        obj = SoMyTest2(name="name", sub=SubObject(subname="subname"))
        assert obj._flatten() == ("name", "subname")
    finally:
        Registry.remove(SubObject)
        Registry.remove(SoMyTest1)
        Registry.remove(SoMyTest2)


def test_delimiter_in_name():
    with pytest.raises(RuntimeError):

        class _MyClass(Feature):
            var__name: str


def test_deserialize_nested():
    class Child(Feature):
        type: int
        name: str = Field(default="test1")

    class Parent(Feature):
        name: str
        child: Child

    in_db_map = {
        "name": "a1",
        "child__type": 42,
        "child__name": "a2",
    }

    p = Parent._unflatten(in_db_map)

    assert p.name == "a1"
    assert p.child.type == 42
    assert p.child.name == "a2"


def test_deserialize_nested_with_name_normalization():
    class ChildClass(Feature):
        type: int
        name: str = Field(default="test1")

    class Parent2(Feature):
        name: str
        childClass11: ChildClass  # noqa: N815

    in_db_map = {
        "name": "name1",
        "child_class11__type": 12,
        "child_class11__name": "n2",
    }

    p = Parent2._unflatten(in_db_map)

    assert p.name == "name1"
    assert p.childClass11.type == 12
    assert p.childClass11.name == "n2"


def test_type_array_of_floats():
    class _Test(Feature):
        d: list[float]

    dict_ = {"d": [1, 3, 5]}
    t = _Test(**dict_)
    assert t.d == [1, 3, 5]


def test_class_attr_resolver_basic():
    class _MyTest(Feature):
        val1: list[float]
        pp: int

    assert _MyTest.val1.name == "val1"
    assert _MyTest.pp.name == "pp"
    assert isinstance(_MyTest.pp.type, Int64)
    assert isinstance(_MyTest.val1.type, Array)


def test_class_attr_resolver_shallow():
    class _MyTest(Feature):
        val1: list[float]
        pp: int

    assert _MyTest.val1.name == "val1"
    assert _MyTest.pp.name == "pp"
    assert isinstance(_MyTest.pp.type, Int64)
    assert isinstance(_MyTest.val1.type, Array)


def test_class_attr_resolver_nested():
    assert MyTest.subClass.type.name == "sub_class__type"
    assert MyTest.subClass.Name.name == "sub_class__name"
    assert isinstance(MyTest.subClass.type.type, Int64)
    assert isinstance(MyTest.subClass.Name.type, String)


def test_class_attr_resolver_nested_3levels():
    class _MyTest1(Feature):
        a: int

    class _MyTest2(Feature):
        b: _MyTest1

    class _MyTest3(Feature):
        c: _MyTest2

    assert _MyTest3.c.b.a.name == "c__b__a"
    assert isinstance(_MyTest3.c.b.a.type, Int64)


def test_class_attr_resolver_partial():
    class _MyTest1(Feature):
        a: str

    class _MyTest2(Feature):
        b: _MyTest1

    class _MyTest3(Feature):
        c: _MyTest2

    assert _MyTest3.c.b.name == "c__b"


def test_pydantic_to_feature():
    class _MyTextBlock(BaseModel):
        id: int
        type: Literal["text"]

    cls = pydantic_to_feature(_MyTextBlock)
    assert Feature.is_feature(cls)

    spec = SignalSchema({"val": cls}).to_udf_spec()
    assert list(spec.keys()) == ["val__id", "val__type"]
    assert list(spec.values()) == [Int64, String]


def test_pydantic_to_feature_nested():
    class _MyTextBlock(BaseModel):
        id: int
        type: Literal["text"]

    class _MyMessage3(BaseModel):
        val1: Optional[str]
        val2: _MyTextBlock

    cls = pydantic_to_feature(_MyMessage3)
    assert Feature.is_feature(cls)

    spec = SignalSchema({"val": cls}).to_udf_spec()
    assert list(spec) == ["val__val1", "val__val2__id", "val__val2__type"]
    assert list(spec.values()) == [String, Int64, String]


def test_unflatten_to_json():
    class _Child(Feature):
        type: int
        name: str = Field(default="test1")

    class _Parent(Feature):
        name: str
        child: _Child

    p = _Parent(name="parent1", child=_Child(type=12, name="child1"))

    flatten = p._flatten()
    assert _Parent._unflatten_to_json(flatten) == {
        "name": "parent1",
        "child": {"type": 12, "name": "child1"},
    }


def test_unflatten_to_json_list():
    class _Child(Feature):
        type: int
        name: str = Field(default="test1")

    class _Parent(Feature):
        name: str
        children: list[_Child]

    p = _Parent(
        name="parent1",
        children=[_Child(type=12, name="child1"), _Child(type=13, name="child2")],
    )

    flatten = p._flatten()
    json = _Parent._unflatten_to_json(flatten)
    assert json == {
        "name": "parent1",
        "children": [{"type": 12, "name": "child1"}, {"type": 13, "name": "child2"}],
    }


def test_unflatten_to_json_dict():
    class _Child(Feature):
        type: int
        address: str = Field(default="test1")

    class _Parent(Feature):
        name: str
        children: dict[str, _Child]

    p = _Parent(
        name="parent1",
        children={
            "child1": _Child(type=12, address="sf"),
            "child2": _Child(type=13, address="nyc"),
        },
    )

    flatten = p._flatten()
    json = _Parent._unflatten_to_json(flatten)
    assert json == {
        "name": "parent1",
        "children": {
            "child1": {"type": 12, "address": "sf"},
            "child2": {"type": 13, "address": "nyc"},
        },
    }


def test_unflatten_to_json_list_of_int():
    class _Child(Feature):
        types: list[int]
        name: str = Field(default="test1")

    child1 = _Child(name="n1", types=[14])
    assert _Child._unflatten_to_json(child1._flatten()) == {"name": "n1", "types": [14]}

    child2 = _Child(name="qwe", types=[1, 2, 3, 5])
    assert _Child._unflatten_to_json(child2._flatten()) == {
        "name": "qwe",
        "types": [1, 2, 3, 5],
    }


def test_unflatten_to_json_list_of_lists():
    class _Child(Feature):
        type: int
        name: str = Field(default="test1")

    class _Parent(Feature):
        name: str
        children: list[_Child]

    class _Company(Feature):
        name: str
        parents: list[_Parent]

    p = _Company(
        name="Co",
        parents=[_Parent(name="parent1", children=[_Child(type=12, name="child1")])],
    )

    assert _Company._unflatten_to_json(p._flatten()) == {
        "name": "Co",
        "parents": [{"name": "parent1", "children": [{"type": 12, "name": "child1"}]}],
    }
