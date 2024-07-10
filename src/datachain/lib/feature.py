import copy
import inspect
import re
import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import lru_cache
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Union,
    get_args,
    get_origin,
)

import attrs
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing_extensions import Literal as LiteralEx

from datachain.lib.feature_registry import Registry
from datachain.query import C
from datachain.query.schema import DEFAULT_DELIMITER
from datachain.sql.types import (
    JSON,
    Array,
    Binary,
    Boolean,
    DateTime,
    Float,
    Int,
    Int32,
    Int64,
    NullType,
    SQLType,
    String,
)

if TYPE_CHECKING:
    from datachain.catalog import Catalog

FeatureStandardType = Union[
    type[int],
    type[str],
    type[float],
    type[bool],
    type[list],
    type[dict],
    type[bytes],
    type[datetime],
]

FeatureType = Union[type["Feature"], FeatureStandardType]
FeatureTypeNames = "Feature, int, str, float, bool, list, dict, bytes, datetime"


TYPE_TO_DATACHAIN = {
    int: Int64,
    str: String,
    Literal: String,
    LiteralEx: String,
    float: Float,
    bool: Boolean,
    datetime: DateTime,  # Note, list of datetime is not supported yet
    bytes: Binary,  # Note, list of bytes is not supported yet
    list: Array,
    dict: JSON,
}

DATACHAIN_TO_TYPE = {
    Int: int,
    Int32: int,
    Int64: int,
    String: str,
    Float: float,
    Boolean: bool,
    DateTime: datetime,
    Binary: bytes,
    Array(NullType): list,
    JSON: dict,
}


NUMPY_TO_DATACHAIN = {
    np.dtype("int8"): Int,
    np.dtype("int16"): Int,
    np.dtype("int32"): Int,
    np.dtype("int64"): Int,
    np.dtype("uint8"): Int,
    np.dtype("uint16"): Int,
    np.dtype("uint32"): Int,
    np.dtype("uint64"): Int,
    np.dtype("float16"): Float,
    np.dtype("float32"): Float,
    np.dtype("float64"): Float,
    np.dtype("object"): String,
    pd.CategoricalDtype(): String,
}


# Disable Pydantic warning, see https://github.com/iterative/dvcx/issues/1285
warnings.filterwarnings(
    "ignore",
    message="Field name .* shadows an attribute in parent",
    category=UserWarning,
)


# Optimization: Store feature classes in this lookup variable so extra checks can be
# skipped within loops.
feature_classes_lookup: dict[type, bool] = {}


class Feature(BaseModel):
    """A base class for defining data classes that serve as inputs and outputs for
    DataChain processing functions like `map()`, `gen()`, etc. Inherits from
    `pydantic`'s BaseModel.
    """

    _is_file: ClassVar[bool] = False
    _version: ClassVar[int] = 1

    @classmethod
    def _is_hidden(cls):
        return cls.__name__.startswith("_")

    def get_value(self, *args: Any, **kwargs: Any) -> Any:
        name = self.__class__.__name__
        raise NotImplementedError(f"get_value() is not defined for feature '{name}'")

    @classmethod
    def _name(cls) -> str:
        return f"{cls.__name__}@{cls._version}"

    @classmethod
    def __pydantic_init_subclass__(cls):
        Registry.add(cls)
        for name, field_info in cls.model_fields.items():
            attr_value = _resolve(cls, name, field_info, prefix=[])
            setattr(cls, name, RestrictedAttribute(attr_value, cls, name))

    @classmethod
    def _prefix(cls) -> str:
        return cls._normalize(cls.__name__)

    @classmethod
    def _normalize(cls, name: str) -> str:
        if DEFAULT_DELIMITER in name:
            raise RuntimeError(
                f"variable '{name}' cannot be used "
                f"because it contains {DEFAULT_DELIMITER}"
            )
        return Feature._to_snake_case(name)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a CamelCase name to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def _set_stream(self, catalog: "Catalog", caching_enabled: bool = False) -> None:
        pass

    @classmethod
    def get_file_signals(cls, path: list[str]) -> Iterable[list[str]]:
        if cls._is_file:
            yield path

        for name, f_info in cls.model_fields.items():
            anno = f_info.annotation
            if Feature.is_feature(anno):
                yield from anno.get_file_signals([*path, name])  # type: ignore[union-attr]

    @classmethod
    def is_feature(cls, anno) -> bool:
        if anno in feature_classes_lookup:
            # Optimization: Skip expensive subclass checks if already checked.
            return feature_classes_lookup[anno]
        is_class = inspect.isclass(anno)
        result = (
            is_class
            and not isinstance(anno, GenericAlias)
            and issubclass(anno, Feature)
        )
        if is_class:
            # Only cache types in the feature classes lookup dict (not instances).
            feature_classes_lookup[anno] = result
        return result

    @classmethod
    def is_standard_type(cls, t: type) -> bool:
        return any(
            t is ft or t is get_args(ft)[0] for ft in get_args(FeatureStandardType)
        )

    @classmethod
    def is_feature_type(cls, t: type) -> bool:
        if cls.is_standard_type(t):
            return True
        if get_origin(t) is list and len(get_args(t)) == 1:
            return cls.is_feature_type(get_args(t)[0])
        return cls.is_feature(t)

    def _flatten_fields_values(self, fields, model):
        for name, f_info in fields.items():
            anno = f_info.annotation
            # Optimization: Access attributes directly to skip the model_dump() call.
            value = getattr(model, name)

            if isinstance(value, list):
                yield [
                    val.model_dump() if Feature.is_feature(type(val)) else val
                    for val in value
                ]
            elif isinstance(value, dict):
                yield {
                    key: val.model_dump() if Feature.is_feature(type(val)) else val
                    for key, val in value.items()
                }
            elif Feature.is_feature(anno):
                yield from self._flatten_fields_values(anno.model_fields, value)
            else:
                yield value

    def _flatten(self):
        return tuple(self._flatten_fields_values(self.model_fields, self))

    @staticmethod
    def _flatten_list(objs):
        return tuple(
            val
            for obj in objs
            for val in obj._flatten_fields_values(obj.model_fields, obj)
        )

    @classmethod
    def _unflatten_with_path(cls, dump, name_path: list[str]):
        res = {}
        for name, f_info in cls.model_fields.items():
            anno = f_info.annotation
            name_norm = cls._normalize(name)
            lst = copy.copy(name_path)

            if inspect.isclass(anno) and issubclass(anno, Feature):
                lst.append(name_norm)
                val = anno._unflatten_with_path(dump, lst)
                res[name] = val
            else:
                lst.append(name_norm)
                curr_path = DEFAULT_DELIMITER.join(lst)
                res[name] = dump[curr_path]
        return cls(**res)

    @classmethod
    def _unflatten(cls, dump):
        return cls._unflatten_with_path(dump, [])

    @classmethod
    def _unflatten_to_json(cls, row: Sequence[Any], pos=0) -> dict:
        return cls._unflatten_to_json_pos(row, pos)[0]

    @classmethod
    def _unflatten_to_json_pos(cls, row: Sequence[Any], pos=0) -> tuple[dict, int]:
        res = {}
        for name, f_info in cls.model_fields.items():
            anno = f_info.annotation
            origin = get_origin(anno)
            if (
                origin not in (list, dict)
                and inspect.isclass(anno)
                and issubclass(anno, Feature)
            ):
                res[name], pos = anno._unflatten_to_json_pos(row, pos)
            else:
                res[name] = row[pos]
                pos += 1
        return res, pos

    @classmethod
    @lru_cache(maxsize=1000)
    def build_tree(cls):
        res = {}

        for name, f_info in cls.model_fields.items():
            anno = f_info.annotation
            subtree = anno.build_tree() if Feature.is_feature(anno) else None
            res[name] = (anno, subtree)

        return res


class RestrictedAttribute:
    """Descriptor implementing an attribute that can only be accessed through
    the defining class and not from subclasses or instances.

    Since it is a non-data descriptor, instance dicts have precedence over it.
    Cannot be used with slotted classes.
    """

    def __init__(self, value, cls=None, name=None):
        self.cls = cls
        self.value = value
        self.name = name

    def __get__(self, instance, owner):
        if owner is not self.cls:
            raise AttributeError(
                f"'{type(owner).__name__}' object has no attribute '{self.name}'"
            )
        if instance is not None:
            raise RuntimeError(
                f"Invalid attempt to access class attribute '{self.name}' through "
                f"'{type(owner).__name__}' instance"
            )
        return self.value

    def __set_name__(self, cls, name):
        self.cls = cls
        self.name = name


@attrs.define
class FeatureAttributeWrapper:
    cls: type[Feature]
    prefix: list[str]

    @property
    def name(self) -> str:
        return DEFAULT_DELIMITER.join(self.prefix)

    def __getattr__(self, name):
        field_info = self.cls.model_fields.get(name)
        if field_info:
            return _resolve(self.cls, name, field_info, prefix=self.prefix)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def _resolve(cls, name, field_info, prefix: list[str]):
    """Resolve feature attributes so they can be used in select(), join()
    and similar functions.

    Users just use `MyClass.sub_attr1.sub_attr2.field` and it will return a DB column
    with a proper name (with default naming - `my_class__sub_attr1__sub_attr2__field`).
    """
    anno = field_info.annotation
    norm_name = cls._normalize(name)

    if not cls.is_feature(anno):
        try:
            anno_sql_class = convert_type_to_datachain(anno)
        except TypeError:
            anno_sql_class = NullType
        new_prefix = copy.copy(prefix)
        new_prefix.append(norm_name)
        return C(DEFAULT_DELIMITER.join(new_prefix), anno_sql_class)

    return FeatureAttributeWrapper(anno, [*prefix, norm_name])


def convert_type_to_datachain(typ):  # noqa: PLR0911
    if inspect.isclass(typ) and issubclass(typ, SQLType):
        return typ

    res = TYPE_TO_DATACHAIN.get(typ)
    if res:
        return res

    orig = get_origin(typ)

    if orig in (Literal, LiteralEx):
        return String

    args = get_args(typ)
    if inspect.isclass(orig) and (issubclass(list, orig) or issubclass(tuple, orig)):
        if args is None or len(args) != 1:
            raise TypeError(f"Cannot resolve type '{typ}' for flattening features")

        args0 = args[0]
        if Feature.is_feature(args0):
            return Array(JSON())

        next_type = convert_type_to_datachain(args0)
        return Array(next_type)

    if inspect.isclass(orig) and issubclass(dict, orig):
        return JSON

    if orig == Union and len(args) == 2 and (type(None) in args):
        return convert_type_to_datachain(args[0])

    # Special case for list in JSON: Union[dict, list[dict]]
    if orig == Union and len(args) >= 2:
        args_no_nones = [arg for arg in args if arg != type(None)]
        if len(args_no_nones) == 2:
            args_no_dicts = [arg for arg in args_no_nones if arg is not dict]
            if len(args_no_dicts) == 1 and get_origin(args_no_dicts[0]) is list:
                arg = get_args(args_no_dicts[0])
                if len(arg) == 1 and arg[0] is dict:
                    return JSON

    raise TypeError(f"Cannot recognize type {typ}")
