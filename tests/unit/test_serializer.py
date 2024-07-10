import base64
import pickle

import pytest

from datachain.data_storage.serializer import Serializable, deserialize


class MySerializableInit(Serializable):
    def __init__(self, name, optional=None):
        self.name = name
        self.optional = optional

    def clone_params(self):
        return MySerializableInit, [self.name], {"optional": self.optional}

    def get_params(self):
        return self.name, self.optional


class MySerializableFunc(Serializable):
    def __init__(self, name, optional=None):
        self.name = name
        self.optional = optional

    @classmethod
    def from_params(cls, name, optional=None):
        return cls(name, optional=optional)

    def clone_params(self):
        return self.from_params, [self.name], {"optional": self.optional}

    def get_params(self):
        return self.name, self.optional


class MySerializableNoParams(Serializable):
    def clone_params(self):
        return MySerializableNoParams, [], {}


@pytest.mark.parametrize(
    "cls,call",
    [
        (MySerializableInit, MySerializableInit),
        (MySerializableFunc, MySerializableFunc.from_params),
    ],
)
@pytest.mark.parametrize(
    "name,optional",
    [
        (None, None),
        ("foo", None),
        (None, 12),
        ("bar", 24),
    ],
)
def test_serializable_init(cls, call, name, optional):
    obj = cls(name, optional=optional)
    assert obj.clone_params() == (call, [name], {"optional": optional})

    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert str(f) == str(call)
    assert args == [name]
    assert kwargs == {"optional": optional}

    obj2 = deserialize(serialized)
    assert isinstance(obj2, cls)
    assert obj2.name == name
    assert obj2.optional == optional
    assert obj2.get_params() == (name, optional)


def test_serializable_init_no_params():
    obj = MySerializableNoParams()
    assert obj.clone_params() == (MySerializableNoParams, [], {})

    serialized = obj.serialize()
    assert serialized
    serialized_pickled = base64.b64decode(serialized.encode())
    assert serialized_pickled
    (f, args, kwargs) = pickle.loads(serialized_pickled)  # noqa: S301
    assert f == MySerializableNoParams
    assert args == []
    assert kwargs == {}

    obj2 = deserialize(serialized)
    assert isinstance(obj2, MySerializableNoParams)
