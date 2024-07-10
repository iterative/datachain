"""
SQL types.

This module provides SQL types to provide common features and interoperability
between different database backends which often have different typing systems.

See https://docs.sqlalchemy.org/en/20/core/custom_types.html#sqlalchemy.types.TypeDecorator.load_dialect_impl

For the corresponding python to db type conversion, it's often simpler and
more direct to use methods at the DBAPI rather than sqlalchemy. For example
for sqlite we can use `sqlite.register_converter`
( https://docs.python.org/3/library/sqlite3.html#sqlite3.register_converter )
"""

from datetime import datetime
from types import MappingProxyType
from typing import Any, Union

from sqlalchemy import TypeDecorator, types

_registry: dict[str, "TypeConverter"] = {}
registry = MappingProxyType(_registry)

_read_converter_registry: dict[str, "TypeReadConverter"] = {}
read_converter_registry = MappingProxyType(_read_converter_registry)

_type_defaults_registry: dict[str, "TypeDefaults"] = {}
type_defaults_registry = MappingProxyType(_type_defaults_registry)

NullType = types.NullType


def register_backend_types(dialect_name: str, type_cls):
    _registry[dialect_name] = type_cls


def register_type_read_converters(dialect_name: str, trc: "TypeReadConverter"):
    _read_converter_registry[dialect_name] = trc


def register_type_defaults(dialect_name: str, td: "TypeDefaults"):
    _type_defaults_registry[dialect_name] = td


def converter(dialect) -> "TypeConverter":
    name = dialect.name
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"No type converter registered for dialect: {dialect.name!r}"
        ) from None


def read_converter(dialect) -> "TypeReadConverter":
    name = dialect.name
    try:
        return read_converter_registry[name]
    except KeyError:
        raise ValueError(
            f"No read type converter registered for dialect: {dialect.name!r}"
        ) from None


def type_defaults(dialect) -> "TypeDefaults":
    name = dialect.name
    try:
        return type_defaults_registry[name]
    except KeyError:
        raise ValueError(f"No type defaults registered for dialect: {name!r}") from None


class SQLType(TypeDecorator):
    impl: type[types.TypeEngine[Any]] = types.TypeEngine
    cache_ok = True

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, _: dict[str, Any]) -> Union[type["SQLType"], "SQLType"]:
        return cls


class String(SQLType):
    impl = types.String

    @property
    def python_type(self):
        return str

    def load_dialect_impl(self, dialect):
        return converter(dialect).string()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).string()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).string(value)


class Boolean(SQLType):
    impl = types.Boolean

    @property
    def python_type(self):
        return bool

    def load_dialect_impl(self, dialect):
        return converter(dialect).boolean()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).boolean()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).boolean(value)


class Int(SQLType):
    impl = types.INTEGER

    @property
    def python_type(self):
        return int

    def load_dialect_impl(self, dialect):
        return converter(dialect).int()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).int()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).int(value)


class Int32(Int):
    def load_dialect_impl(self, dialect):
        return converter(dialect).int32()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).int32()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).int32(value)


class Int64(Int):
    def load_dialect_impl(self, dialect):
        return converter(dialect).int64()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).int64()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).int64(value)


class UInt64(Int):
    def load_dialect_impl(self, dialect):
        return converter(dialect).uint64()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).uint64()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).uint64(value)


class Float(SQLType):
    impl = types.INTEGER

    @property
    def python_type(self):
        return float

    def load_dialect_impl(self, dialect):
        return converter(dialect).float()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).float()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).float(value)


class Float32(Float):
    def load_dialect_impl(self, dialect):
        return converter(dialect).float32()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).float32()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).float32(value)


class Float64(Float):
    def load_dialect_impl(self, dialect):
        return converter(dialect).float64()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).float64()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).float64(value)


class Array(SQLType):
    impl = types.ARRAY

    @property
    def python_type(self):
        return list

    def load_dialect_impl(self, dialect):
        return converter(dialect).array(self.item_type)

    def to_dict(self) -> dict[str, Any]:
        item_type_dict = (
            self.item_type.to_dict()
            if isinstance(self.item_type, SQLType)
            else self.item_type().to_dict()
        )
        return {
            "type": self.__class__.__name__,
            "item_type": item_type_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Union[type["SQLType"], "SQLType"]:
        sub_t = NAME_TYPES_MAPPING[d["item_type"]["type"]].from_dict(  # type: ignore [attr-defined]
            d["item_type"]
        )
        return cls(sub_t)

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).array()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).array(value, self.item_type, dialect)


class JSON(SQLType):
    impl = types.JSON

    @property
    def python_type(self):
        return dict

    def load_dialect_impl(self, dialect):
        return converter(dialect).json()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).json()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).json(value)


class DateTime(SQLType):
    impl = types.DATETIME

    @property
    def python_type(self):
        return datetime

    def load_dialect_impl(self, dialect):
        return converter(dialect).datetime()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).datetime()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).datetime(value)


class Binary(SQLType):
    impl = types.BINARY

    @property
    def python_type(self):
        return bytes

    def load_dialect_impl(self, dialect):
        return converter(dialect).binary()

    @staticmethod
    def default_value(dialect):
        return type_defaults(dialect).binary()

    def on_read_convert(self, value, dialect):
        return read_converter(dialect).binary(value)


class TypeReadConverter:
    def string(self, value):
        return value

    def boolean(self, value):
        return value

    def int(self, value):
        return value

    def int32(self, value):
        return value

    def int64(self, value):
        return value

    def uint64(self, value):
        return value

    def float(self, value):
        return value

    def float32(self, value):
        return value

    def float64(self, value):
        return value

    def array(self, value, item_type, dialect):
        if value is None or item_type is None:
            return value
        return [item_type.on_read_convert(x, dialect) for x in value]

    def json(self, value):
        return value

    def datetime(self, value):
        return value

    def uuid(self, value):
        return value

    def binary(self, value):
        return value


class TypeConverter:
    def string(self):
        return types.String()

    def boolean(self):
        return types.Boolean()

    def int(self):
        return types.Integer()

    def int32(self):
        return self.int()

    def int64(self):
        return self.int()

    def uint64(self):
        return self.int()

    def float(self):
        return types.Float()

    def float32(self):
        return self.float()

    def float64(self):
        return self.float()

    def array(self, item_type):
        return types.ARRAY(item_type)

    def json(self):
        return types.JSON()

    def datetime(self):
        return types.DATETIME()

    def binary(self):
        return types.BINARY()


class TypeDefaults:
    def string(self):
        return None

    def boolean(self):
        return None

    def int(self):
        return None

    def int32(self):
        return None

    def int64(self):
        return None

    def uint64(self):
        return None

    def float(self):
        return None

    def float32(self):
        return None

    def float64(self):
        return None

    def array(self):
        return None

    def json(self):
        return None

    def datetime(self):
        return None

    def uuid(self):
        return None

    def binary(self):
        return None


TYPES = [
    String,
    Boolean,
    Int,
    Int32,
    Int64,
    UInt64,
    Float,
    Float32,
    Float64,
    Array,
    JSON,
    DateTime,
    Binary,
]

NAME_TYPES_MAPPING = {t.__name__: t for t in TYPES}
