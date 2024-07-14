from typing import ClassVar, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from datachain.lib.feature import ModelUtil, VersionedModel, is_feature
from datachain.lib.feature_registry import Registry
from datachain.lib.feature_utils import pydantic_to_feature
from datachain.lib.signal_schema import SignalSchema
from datachain.sql.types import (
    Int64,
    String,
)


class FileBasic(VersionedModel):
    parent: str = Field(default="")
    name: str
    size: int = Field(default=0)


class TestFileInfo(FileBasic):
    location: dict = Field(default={})


class FileInfoEx(VersionedModel):
    f_info: TestFileInfo
    type_id: int


class MyNestedClass(VersionedModel):
    type: int
    Name: str = Field(default="test1")


class MyTest(VersionedModel):
    ThisIsName: str
    subClass: MyNestedClass  # noqa: N815


def test_flatten_basic():
    vals = ModelUtil.flatten(FileBasic(parent="hello", name="world", size=123))
    assert vals == ("hello", "world", 123)


def test_flatten_with_json():
    t1 = TestFileInfo(parent="prt4", name="test1", size=42, location={"ee": "rr"})
    assert ModelUtil.flatten(t1) == ("prt4", "test1", 42, {"ee": "rr"})


def test_flatten_with_empty_json():
    with pytest.raises(ValidationError):
        TestFileInfo(parent="prt4", name="test1", size=42, location=None)


def test_flatten_with_accepted_empty_json():
    class _Test(VersionedModel):
        d: Optional[dict]

    assert ModelUtil.flatten(_Test(d=None)) == (None,)


def test_flatten_nested():
    t0 = TestFileInfo(parent="sfo", name="sf", size=567, location={"42": 999})
    t1 = FileInfoEx(f_info=t0, type_id=1849)

    assert ModelUtil.flatten(t1) == ("sfo", "sf", 567, {"42": 999}, 1849)


def test_flatten_list():
    t1 = TestFileInfo(parent="p1", name="n4", size=3, location={"a": "b"})
    t2 = TestFileInfo(parent="p2", name="n5", size=2, location={"c": "d"})

    vals = ModelUtil.flatten_list([t1, t2])
    assert vals == ("p1", "n4", 3, {"a": "b"}, "p2", "n5", 2, {"c": "d"})


def test_registry():
    class MyTestRndmz(VersionedModel):
        name: str
        count: int

    Registry.add(MyTestRndmz)
    assert Registry.get(MyTestRndmz.__name__) == MyTestRndmz
    assert Registry.get(MyTestRndmz.__name__, version=1) == MyTestRndmz
    Registry.remove(MyTestRndmz)


def test_registry_versioned():
    class MyTestXYZ(VersionedModel):
        _version: ClassVar[int] = 42
        name: str
        count: int

    assert Registry.get(MyTestXYZ.__name__) == MyTestXYZ
    assert Registry.get(MyTestXYZ.__name__, version=1) is None
    assert Registry.get(MyTestXYZ.__name__, version=42) == MyTestXYZ
    Registry.remove(MyTestXYZ)


def test_inheritance():
    class SubObject(VersionedModel):
        subname: str

    class SoMyTest1(VersionedModel):
        name: str
        sub: SubObject

    class SoMyTest2(SoMyTest1):
        pass

    try:
        with pytest.raises(ValueError):
            SoMyTest2()

        obj = SoMyTest2(name="name", sub=SubObject(subname="subname"))
        assert ModelUtil.flatten(obj) == ("name", "subname")
    finally:
        Registry.remove(SubObject)
        Registry.remove(SoMyTest1)
        Registry.remove(SoMyTest2)


def test_deserialize_nested():
    class Child(VersionedModel):
        type: int
        name: str = Field(default="test1")

    class Parent(VersionedModel):
        name: str
        child: Child

    in_db_map = {
        "name": "a1",
        "child__type": 42,
        "child__name": "a2",
    }

    p = ModelUtil.unflatten(Parent, in_db_map)

    assert p.name == "a1"
    assert p.child.type == 42
    assert p.child.name == "a2"


def test_deserialize_nested_with_name_normalization():
    class ChildClass(VersionedModel):
        type: int
        name: str = Field(default="test1")

    class Parent2(VersionedModel):
        name: str
        childClass11: ChildClass  # noqa: N815

    in_db_map = {
        "name": "name1",
        "child_class11__type": 12,
        "child_class11__name": "n2",
    }

    p = ModelUtil.unflatten(Parent2, in_db_map)

    assert p.name == "name1"
    assert p.childClass11.type == 12
    assert p.childClass11.name == "n2"


def test_type_array_of_floats():
    class _Test(VersionedModel):
        d: list[float]

    dict_ = {"d": [1, 3, 5]}
    t = _Test(**dict_)
    assert t.d == [1, 3, 5]


def test_pydantic_to_feature():
    class _MyTextBlock(BaseModel):
        id: int
        type: Literal["text"]

    cls = pydantic_to_feature(_MyTextBlock)
    assert is_feature(cls)

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
    assert is_feature(cls)

    spec = SignalSchema({"val": cls}).to_udf_spec()
    assert list(spec) == ["val__val1", "val__val2__id", "val__val2__type"]
    assert list(spec.values()) == [String, Int64, String]


def test_unflatten_to_json():
    class _Child(VersionedModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(VersionedModel):
        name: str
        child: _Child

    p = _Parent(name="parent1", child=_Child(type=12, name="child1"))

    flatten = ModelUtil.flatten(p)
    assert ModelUtil.unflatten_to_json(_Parent, flatten) == {
        "name": "parent1",
        "child": {"type": 12, "name": "child1"},
    }


def test_unflatten_to_json_list():
    class _Child(VersionedModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(VersionedModel):
        name: str
        children: list[_Child]

    p = _Parent(
        name="parent1",
        children=[_Child(type=12, name="child1"), _Child(type=13, name="child2")],
    )

    flatten = ModelUtil.flatten(p)
    json = ModelUtil.unflatten_to_json(_Parent, flatten)
    assert json == {
        "name": "parent1",
        "children": [{"type": 12, "name": "child1"}, {"type": 13, "name": "child2"}],
    }


def test_unflatten_to_json_dict():
    class _Child(VersionedModel):
        type: int
        address: str = Field(default="test1")

    class _Parent(VersionedModel):
        name: str
        children: dict[str, _Child]

    p = _Parent(
        name="parent1",
        children={
            "child1": _Child(type=12, address="sf"),
            "child2": _Child(type=13, address="nyc"),
        },
    )

    flatten = ModelUtil.flatten(p)
    json = ModelUtil.unflatten_to_json(_Parent, flatten)
    assert json == {
        "name": "parent1",
        "children": {
            "child1": {"type": 12, "address": "sf"},
            "child2": {"type": 13, "address": "nyc"},
        },
    }


def test_unflatten_to_json_list_of_int():
    class _Child(VersionedModel):
        types: list[int]
        name: str = Field(default="test1")

    child1 = _Child(name="n1", types=[14])
    assert ModelUtil.unflatten_to_json(_Child, ModelUtil.flatten(child1)) == {
        "name": "n1",
        "types": [14],
    }

    child2 = _Child(name="qwe", types=[1, 2, 3, 5])
    assert ModelUtil.unflatten_to_json(_Child, ModelUtil.flatten(child2)) == {
        "name": "qwe",
        "types": [1, 2, 3, 5],
    }


def test_unflatten_to_json_list_of_lists():
    class _Child(VersionedModel):
        type: int
        name: str = Field(default="test1")

    class _Parent(VersionedModel):
        name: str
        children: list[_Child]

    class _Company(VersionedModel):
        name: str
        parents: list[_Parent]

    p = _Company(
        name="Co",
        parents=[_Parent(name="parent1", children=[_Child(type=12, name="child1")])],
    )

    assert ModelUtil.unflatten_to_json(_Company, ModelUtil.flatten(p)) == {
        "name": "Co",
        "parents": [{"name": "parent1", "children": [{"type": 12, "name": "child1"}]}],
    }


def test_version():
    class _MyCls(BaseModel):
        name: str
        age: int
        _version: ClassVar[int] = 23

    assert Registry.get_version(_MyCls) == 23
