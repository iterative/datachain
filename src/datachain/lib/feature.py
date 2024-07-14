import copy
import inspect
import re
import warnings
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from typing_extensions import Literal as LiteralEx

from datachain.lib.feature_registry import Registry
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

FeatureType = Union[type[BaseModel], FeatureStandardType]
FeatureTypeNames = "BaseModel, int, str, float, bool, list, dict, bytes, datetime"

TYPE_TO_DATACHAIN = {
    int: Int64,
    str: String,
    Literal: String,
    LiteralEx: String,
    Enum: String,
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


# Disable Pydantic warning, see https://github.com/iterative/dvcx/issues/1285
warnings.filterwarnings(
    "ignore",
    message="Field name .* shadows an attribute in parent",
    category=UserWarning,
)

# Optimization: Store feature classes in this lookup variable so extra checks can be
# skipped within loops.
feature_classes_lookup: dict[type, bool] = {}


def registry_list(output: Sequence[FeatureType]):
    for val in output:
        if is_feature(val):
            Registry.add(val)


def is_standard_type(t: type) -> bool:
    return any(t is ft or t is get_args(ft)[0] for ft in get_args(FeatureStandardType))


def is_feature_type(t: type) -> bool:
    if is_standard_type(t):
        return True
    if get_origin(t) is list and len(get_args(t)) == 1:
        return is_feature_type(get_args(t)[0])
    return issubclass(t, BaseModel)


def is_feature(anno) -> bool:
    if anno in feature_classes_lookup:
        # Optimization: Skip expensive subclass checks if already checked.
        return feature_classes_lookup[anno]
    is_class = inspect.isclass(anno)
    result = (
        is_class and not isinstance(anno, GenericAlias) and issubclass(anno, BaseModel)
    )
    if is_class:
        # Only cache types in the feature classes lookup dict (not instances).
        feature_classes_lookup[anno] = result
    return result


def to_feature(typ: Optional[Any]) -> Optional[type[BaseModel]]:
    if typ is None or not is_feature(typ):
        return None
    return typ


def build_tree(model: Optional[Any]):
    if model is None:
        return None
    if (fr := to_feature(model)) is not None:
        return _build_tree(fr)
    return None


def _build_tree(model: type[BaseModel]):
    res = {}

    for name, f_info in model.model_fields.items():
        anno = f_info.annotation
        if (fr := to_feature(anno)) is not None:
            subtree = build_tree(fr)
        else:
            subtree = None
        res[name] = (anno, subtree)

    return res


class VersionedModel(BaseModel):
    _version: ClassVar[int] = 1

    def get_value(self):
        return None

    @classmethod
    def __pydantic_init_subclass__(cls):
        Registry.add(cls)


class FileFeature(VersionedModel):
    def _set_stream(self, catalog: "Catalog", caching_enabled: bool = False) -> None:
        pass

    def open(self):
        raise NotImplementedError

    def read(self):
        with self.open() as stream:
            return stream.read()

    def get_value(self):
        return self.read()


class ModelUtil:
    @classmethod
    def flatten(cls, obj: BaseModel):
        return tuple(cls._flatten_fields_values(obj.model_fields, obj))

    @classmethod
    def _flatten_fields_values(cls, fields, obj: BaseModel):
        for name, f_info in fields.items():
            anno = f_info.annotation
            # Optimization: Access attributes directly to skip the model_dump() call.
            value = getattr(obj, name)

            if isinstance(value, list):
                yield [
                    val.model_dump() if is_feature(type(val)) else val for val in value
                ]
            elif isinstance(value, dict):
                yield {
                    key: val.model_dump() if is_feature(type(val)) else val
                    for key, val in value.items()
                }
            elif is_feature(anno):
                yield from cls._flatten_fields_values(anno.model_fields, value)
            else:
                yield value

    @classmethod
    def unflatten_to_json(
        cls, model: type[BaseModel], row: Sequence[Any], pos=0
    ) -> dict:
        return cls.unflatten_to_json_pos(model, row, pos)[0]

    @classmethod
    def unflatten_to_json_pos(
        cls, model: type[BaseModel], row: Sequence[Any], pos=0
    ) -> tuple[dict, int]:
        res = {}
        for name, f_info in model.model_fields.items():
            anno = f_info.annotation
            origin = get_origin(anno)
            if (
                origin not in (list, dict)
                and inspect.isclass(anno)
                and issubclass(anno, BaseModel)
            ):
                res[name], pos = cls.unflatten_to_json_pos(anno, row, pos)
            else:
                res[name] = row[pos]
                pos += 1
        return res, pos

    #################
    @classmethod
    def _flatten(cls, obj):
        return tuple(cls._flatten_fields_values(obj.model_fields, obj))

    @classmethod
    def flatten_list(cls, obj_list):
        return tuple(
            val
            for obj in obj_list
            for val in cls._flatten_fields_values(obj.model_fields, obj)
        )

    ##################  UNFLATTEN:
    @classmethod
    def _normalize(cls, name: str) -> str:
        if DEFAULT_DELIMITER in name:
            raise RuntimeError(
                f"variable '{name}' cannot be used "
                f"because it contains {DEFAULT_DELIMITER}"
            )
        return cls._to_snake_case(name)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a CamelCase name to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    @classmethod
    def _unflatten_with_path(cls, model: type[BaseModel], dump, name_path: list[str]):
        res = {}
        for name, f_info in model.model_fields.items():
            anno = f_info.annotation
            name_norm = cls._normalize(name)
            lst = copy.copy(name_path)

            if inspect.isclass(anno) and issubclass(anno, BaseModel):
                lst.append(name_norm)
                val = cls._unflatten_with_path(anno, dump, lst)
                res[name] = val
            else:
                lst.append(name_norm)
                curr_path = DEFAULT_DELIMITER.join(lst)
                res[name] = dump[curr_path]
        return model(**res)

    @classmethod
    def unflatten(cls, model: type[BaseModel], dump):
        return cls._unflatten_with_path(model, dump, [])


def convert_type_to_datachain(typ):  # noqa: PLR0911
    if inspect.isclass(typ):
        if issubclass(typ, SQLType):
            return typ
        if issubclass(typ, Enum):
            return str

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
        if is_feature(args0):
            return Array(JSON())

        next_type = convert_type_to_datachain(args0)
        return Array(next_type)

    if orig is Annotated:
        # Ignoring annotations
        return convert_type_to_datachain(args[0])

    if inspect.isclass(orig) and issubclass(dict, orig):
        return JSON

    if orig == Union:
        if len(args) == 2 and (type(None) in args):
            return convert_type_to_datachain(args[0])

        if _is_json_inside_union(orig, args):
            return JSON

    raise TypeError(f"Cannot recognize type {typ}")


def _is_json_inside_union(orig, args) -> bool:
    if orig == Union and len(args) >= 2:
        # List in JSON: Union[dict, list[dict]]
        args_no_nones = [arg for arg in args if arg != type(None)]
        if len(args_no_nones) == 2:
            args_no_dicts = [arg for arg in args_no_nones if arg is not dict]
            if len(args_no_dicts) == 1 and get_origin(args_no_dicts[0]) is list:
                arg = get_args(args_no_dicts[0])
                if len(arg) == 1 and arg[0] is dict:
                    return True

        # List of objects: Union[MyClass, OtherClass]
        if any(inspect.isclass(arg) and issubclass(arg, BaseModel) for arg in args):
            return True
    return False
