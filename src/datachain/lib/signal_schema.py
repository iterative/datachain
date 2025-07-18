import copy
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from inspect import isclass
from typing import (  # noqa: UP035
    IO,
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, create_model
from sqlalchemy import ColumnElement
from typing_extensions import Literal as LiteralEx

from datachain.func import literal
from datachain.func.func import Func
from datachain.lib.convert.python_to_sql import python_to_sql
from datachain.lib.convert.sql_to_python import sql_to_python
from datachain.lib.convert.unflatten import unflatten_to_json_pos
from datachain.lib.data_model import DataModel, DataType, DataValue
from datachain.lib.file import File
from datachain.lib.model_store import ModelStore
from datachain.lib.utils import DataChainParamsError
from datachain.query.schema import DEFAULT_DELIMITER, Column
from datachain.sql.types import SQLType

if TYPE_CHECKING:
    from datachain.catalog import Catalog


NAMES_TO_TYPES = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "bytes": bytes,
    "datetime": datetime,
    "Final": Final,
    "Union": Union,
    "Optional": Optional,
    "List": list,
    "Dict": dict,
    "Literal": Any,
    "Any": Any,
}


class SignalSchemaError(DataChainParamsError):
    pass


class SignalSchemaWarning(RuntimeWarning):
    pass


class SignalResolvingError(SignalSchemaError):
    def __init__(self, path: Optional[list[str]], msg: str):
        name = " '" + ".".join(path) + "'" if path else ""
        super().__init__(f"cannot resolve signal name{name}: {msg}")


class SetupError(SignalSchemaError):
    def __init__(self, name: str, msg: str):
        super().__init__(f"cannot setup value '{name}': {msg}")


class SignalResolvingTypeError(SignalResolvingError):
    def __init__(self, method: str, field):
        super().__init__(
            None,
            f"{method} supports only `str` type"
            f" while '{field}' has type '{type(field)}'",
        )


class SignalRemoveError(SignalSchemaError):
    def __init__(self, path: Optional[list[str]], msg: str):
        name = " '" + ".".join(path) + "'" if path else ""
        super().__init__(f"cannot remove signal name{name}: {msg}")


class CustomType(BaseModel):
    schema_version: int = Field(ge=1, le=2, strict=True)
    name: str
    fields: dict[str, str]
    bases: list[tuple[str, str, Optional[str]]]
    hidden_fields: Optional[list[str]] = None

    @classmethod
    def deserialize(cls, data: dict[str, Any], type_name: str) -> "CustomType":
        version = data.get("schema_version", 1)

        if version == 1:
            data = {
                "schema_version": 1,
                "name": type_name,
                "fields": data,
                "bases": [],
                "hidden_fields": [],
            }

        return cls(**data)


def create_feature_model(
    name: str,
    fields: Mapping[str, Union[type, None, tuple[type, Any]]],
    base: Optional[type] = None,
) -> type[BaseModel]:
    """
    This gets or returns a dynamic feature model for use in restoring a model
    from the custom_types stored within a serialized SignalSchema. This is useful
    when using a custom feature model where the original definition is not available.
    This happens in Studio and if a custom model is used in a dataset, then that dataset
    is used in a DataChain in a separate script where that model is not declared.
    """
    name = name.replace("@", "_")
    return create_model(
        name,
        __base__=base or DataModel,  # type: ignore[call-overload]
        # These are tuples for each field of: annotation, default (if any)
        **{
            field_name: anno if isinstance(anno, tuple) else (anno, None)
            for field_name, anno in fields.items()
        },
    )


@dataclass
class SignalSchema:
    values: dict[str, DataType]
    tree: dict[str, Any]
    setup_func: dict[str, Callable]
    setup_values: Optional[dict[str, Any]]

    def __init__(
        self,
        values: dict[str, DataType],
        setup: Optional[dict[str, Callable]] = None,
    ):
        self.values = values
        self.tree = self._build_tree(values)

        self.setup_func = setup or {}
        self.setup_values = None
        for key, func in self.setup_func.items():
            if not callable(func):
                raise SetupError(key, "value must be function or callable class")

    def _init_setup_values(self) -> None:
        if self.setup_values is not None:
            return

        res = {}
        for key, func in self.setup_func.items():
            try:
                res[key] = func()
            except Exception as ex:
                raise SetupError(key, f"error when call function: '{ex}'") from ex
        self.setup_values = res

    @staticmethod
    def from_column_types(col_types: dict[str, Any]) -> "SignalSchema":
        signals: dict[str, DataType] = {}
        for field, col_type in col_types.items():
            if isinstance(col_type, SQLType):
                signals[field] = col_type.python_type
            elif isclass(col_type) and issubclass(col_type, SQLType):
                signals[field] = col_type().python_type
            else:
                raise SignalSchemaError(
                    f"signal schema cannot be obtained for column '{field}':"
                    f" unsupported type '{col_type}'"
                )
        return SignalSchema(signals)

    @staticmethod
    def _get_bases(fr: type) -> list[tuple[str, str, Optional[str]]]:
        bases: list[tuple[str, str, Optional[str]]] = []
        for base in fr.__mro__:
            model_store_name = (
                ModelStore.get_name(base) if issubclass(base, DataModel) else None
            )
            bases.append((base.__name__, base.__module__, model_store_name))
        return bases

    @staticmethod
    def _serialize_custom_model(
        version_name: str, fr: type[BaseModel], custom_types: dict[str, Any]
    ) -> str:
        """This serializes any custom type information to the provided custom_types
        dict, and returns the name of the type serialized."""
        if version_name in custom_types:
            # This type is already stored in custom_types.
            return version_name
        fields = {}

        for field_name, info in fr.model_fields.items():
            field_type = info.annotation
            # All fields should be typed.
            assert field_type
            fields[field_name] = SignalSchema._serialize_type(field_type, custom_types)

        bases = SignalSchema._get_bases(fr)

        ct = CustomType(
            schema_version=2,
            name=version_name,
            fields=fields,
            bases=bases,
            hidden_fields=getattr(fr, "_hidden_fields", []),
        )
        custom_types[version_name] = ct.model_dump()

        return version_name

    @staticmethod
    def _serialize_type(fr: type, custom_types: dict[str, Any]) -> str:
        """Serialize a given type to a string, including automatic ModelStore
        registration, and save this type and subtypes to custom_types as well."""
        subtypes: list[Any] = []
        type_name = SignalSchema._type_to_str(fr, subtypes)
        # Iterate over all subtypes (includes the input type).
        for st in subtypes:
            if st is None or not ModelStore.is_pydantic(st):
                continue
            # Register and save feature types.
            st_version_name = ModelStore.get_name(st)
            if st is fr:
                # If the main type is Pydantic, then use the ModelStore version name.
                type_name = st_version_name
            # Save this type to custom_types.
            SignalSchema._serialize_custom_model(st_version_name, st, custom_types)
        return type_name

    def serialize(self) -> dict[str, Any]:
        signals: dict[str, Any] = {}
        custom_types: dict[str, Any] = {}
        for name, fr_type in self.values.items():
            signals[name] = self._serialize_type(fr_type, custom_types)
        if custom_types:
            signals["_custom_types"] = custom_types
        return signals

    @staticmethod
    def _split_subtypes(type_name: str) -> list[str]:
        """This splits a list of subtypes, including proper square bracket handling."""
        start = 0
        depth = 0
        subtypes = []
        for i, c in enumerate(type_name):
            if c == "[":
                depth += 1
            elif c == "]":
                if depth == 0:
                    raise ValueError(
                        "Extra closing square bracket when parsing subtype list"
                    )
                depth -= 1
            elif c == "," and depth == 0:
                subtypes.append(type_name[start:i].strip())
                start = i + 1
        if depth > 0:
            raise ValueError("Unclosed square bracket when parsing subtype list")
        subtypes.append(type_name[start:].strip())
        return subtypes

    @staticmethod
    def _deserialize_custom_type(
        type_name: str, custom_types: dict[str, Any]
    ) -> Optional[type]:
        """Given a type name like MyType@v1 gets a type from ModelStore or recreates
        it based on the information from the custom types dict that includes fields and
        bases."""
        model_name, version = ModelStore.parse_name_version(type_name)
        fr = ModelStore.get(model_name, version)
        if fr:
            return fr

        if type_name in custom_types:
            ct = CustomType.deserialize(custom_types[type_name], type_name)

            fields = {
                field_name: SignalSchema._resolve_type(field_type_str, custom_types)
                for field_name, field_type_str in ct.fields.items()
            }

            base_model = None
            for base in ct.bases:
                _, _, model_store_name = base
                if model_store_name:
                    model_name, version = ModelStore.parse_name_version(
                        model_store_name
                    )
                    base_model = ModelStore.get(model_name, version)
                    if base_model:
                        break

            return create_feature_model(type_name, fields, base=base_model)

        return None

    @staticmethod
    def _resolve_type(type_name: str, custom_types: dict[str, Any]) -> Optional[type]:
        """Convert a string-based type back into a python type."""
        type_name = type_name.strip()
        if not type_name:
            raise ValueError("Type cannot be empty")
        if type_name == "NoneType":
            return None

        bracket_idx = type_name.find("[")
        subtypes: Optional[tuple[Optional[type], ...]] = None
        if bracket_idx > -1:
            if bracket_idx == 0:
                raise ValueError("Type cannot start with '['")
            close_bracket_idx = type_name.rfind("]")
            if close_bracket_idx == -1:
                raise ValueError("Unclosed square bracket when parsing type")
            if close_bracket_idx < bracket_idx:
                raise ValueError("Square brackets are out of order when parsing type")
            if close_bracket_idx == bracket_idx + 1:
                raise ValueError("Empty square brackets when parsing type")
            subtype_names = SignalSchema._split_subtypes(
                type_name[bracket_idx + 1 : close_bracket_idx]
            )
            # Types like Union require the parameters to be a tuple of types.
            subtypes = tuple(
                SignalSchema._resolve_type(st, custom_types) for st in subtype_names
            )
            type_name = type_name[:bracket_idx].strip()

        fr = NAMES_TO_TYPES.get(type_name)
        if fr:
            if subtypes:
                if len(subtypes) == 1:
                    # Types like Optional require there to be only one argument.
                    return fr[subtypes[0]]  # type: ignore[index]
                # Other types like Union require the parameters to be a tuple of types.
                return fr[subtypes]  # type: ignore[index]
            return fr  # type: ignore[return-value]

        fr = SignalSchema._deserialize_custom_type(type_name, custom_types)
        if fr:
            return fr

        # This can occur if a third-party or custom type is used, which is not available
        # when deserializing.
        warnings.warn(
            f"Could not resolve type: '{type_name}'.",
            SignalSchemaWarning,
            stacklevel=2,
        )
        return Any  # type: ignore[return-value]

    @staticmethod
    def deserialize(schema: dict[str, Any]) -> "SignalSchema":
        if not isinstance(schema, dict):
            raise SignalSchemaError(f"cannot deserialize signal schema: {schema}")

        signals: dict[str, DataType] = {}
        custom_types: dict[str, Any] = schema.get("_custom_types", {})
        for signal, type_name in schema.items():
            if signal == "_custom_types":
                # This entry is used as a lookup for custom types,
                # and is not an actual field.
                continue
            if not isinstance(type_name, str):
                raise SignalSchemaError(
                    f"cannot deserialize '{type_name}': "
                    "serialized types must be a string"
                )
            try:
                fr = SignalSchema._resolve_type(type_name, custom_types)
                if fr is Any:
                    # Skip if the type is not found, so all data can be displayed.
                    warnings.warn(
                        f"In signal '{signal}': "
                        f"unknown type '{type_name}'."
                        f" Try to add it with `ModelStore.register({type_name})`.",
                        SignalSchemaWarning,
                        stacklevel=2,
                    )
                    continue
            except ValueError as err:
                raise SignalSchemaError(
                    f"cannot deserialize '{signal}': {err}"
                ) from err
            signals[signal] = fr  # type: ignore[assignment]

        return SignalSchema(signals)

    @staticmethod
    def get_flatten_hidden_fields(schema: dict):
        custom_types = schema.get("_custom_types", {})
        if not custom_types:
            return []

        hidden_by_types = {
            name: schema.get("hidden_fields", [])
            for name, schema in custom_types.items()
        }

        hidden_fields = []

        def traverse(prefix, schema_info):
            for field, field_type in schema_info.items():
                if field == "_custom_types":
                    continue

                if field_type in custom_types:
                    hidden_fields.extend(
                        f"{prefix}{field}__{f}" for f in hidden_by_types[field_type]
                    )
                    traverse(
                        prefix + field + "__",
                        custom_types[field_type].get("fields", {}),
                    )

        traverse("", schema)

        return hidden_fields

    def to_udf_spec(self) -> dict[str, type]:
        res = {}
        for path, type_, has_subtree, _ in self.get_flat_tree():
            if path[0] in self.setup_func:
                continue
            if not has_subtree:
                db_name = DEFAULT_DELIMITER.join(path)
                res[db_name] = python_to_sql(type_)
        return res

    def row_to_objs(self, row: Sequence[Any]) -> list[Any]:
        self._init_setup_values()

        objs: list[Any] = []
        pos = 0
        for name, fr_type in self.values.items():
            if self.setup_values and name in self.setup_values:
                objs.append(self.setup_values.get(name))
            elif (fr := ModelStore.to_pydantic(fr_type)) is not None:
                j, pos = unflatten_to_json_pos(fr, row, pos)
                objs.append(fr(**j))
            else:
                objs.append(row[pos])
                pos += 1
        return objs

    def get_file_signal(self) -> Optional[str]:
        for signal_name, signal_type in self.values.items():
            if (fr := ModelStore.to_pydantic(signal_type)) is not None and issubclass(
                fr, File
            ):
                return signal_name
        return None

    def slice(
        self,
        params: dict[str, Union[DataType, Any]],
        setup: Optional[dict[str, Callable]] = None,
        is_batch: bool = False,
    ) -> "SignalSchema":
        """
        Returns new schema that combines current schema and setup signals.
        """
        setup_params = setup.keys() if setup else []
        schema: dict[str, DataType] = {}

        for param, param_type in params.items():
            # This is special case for setup params, they are always treated as strings
            if param in setup_params:
                schema[param] = str
                continue

            schema_type = self._find_in_tree(param.split("."))

            if param_type is Any:
                schema[param] = schema_type
                continue

            schema_origin = get_origin(schema_type)
            param_origin = get_origin(param_type)

            if schema_origin is Union and type(None) in get_args(schema_type):
                schema_type = get_args(schema_type)[0]
                if param_origin is Union and type(None) in get_args(param_type):
                    param_type = get_args(param_type)[0]

            if is_batch:
                if param_type is list:
                    schema[param] = schema_type
                    continue

                if param_origin is not list:
                    raise SignalResolvingError(param.split("."), "is not a list")

                param_type = get_args(param_type)[0]

            if param_type == schema_type or (
                isclass(param_type)
                and isclass(schema_type)
                and issubclass(param_type, File)
                and issubclass(schema_type, File)
            ):
                schema[param] = schema_type
                continue

            raise SignalResolvingError(
                param.split("."),
                f"types mismatch: {param_type} != {schema_type}",
            )

        return SignalSchema(schema, setup)

    def row_to_features(
        self, row: Sequence, catalog: "Catalog", cache: bool = False
    ) -> list[DataValue]:
        res = []
        pos = 0
        for fr_cls in self.values.values():
            if (fr := ModelStore.to_pydantic(fr_cls)) is None:
                res.append(row[pos])
                pos += 1
            else:
                json, pos = unflatten_to_json_pos(fr, row, pos)  # type: ignore[union-attr]
                obj = fr(**json)
                SignalSchema._set_file_stream(obj, catalog, cache)
                res.append(obj)
        return res

    @staticmethod
    def _set_file_stream(
        obj: BaseModel, catalog: "Catalog", cache: bool = False
    ) -> None:
        if isinstance(obj, File):
            obj._set_stream(catalog, caching_enabled=cache)
        for field, finfo in type(obj).model_fields.items():
            if ModelStore.is_pydantic(finfo.annotation):
                SignalSchema._set_file_stream(getattr(obj, field), catalog, cache)

    def get_column_type(self, col_name: str, with_subtree: bool = False) -> DataType:
        """
        Returns column type by column name.

        If `with_subtree` is True, then it will return the type of the column
        even if it has a subtree (e.g. model with nested fields), otherwise it will
        return the type of the column (standard type field, not the model).

        If column is not found, raises `SignalResolvingError`.
        """
        for path, _type, has_subtree, _ in self.get_flat_tree():
            if (with_subtree or not has_subtree) and DEFAULT_DELIMITER.join(
                path
            ) == col_name:
                return _type
        raise SignalResolvingError([col_name], "is not found")

    def db_signals(
        self, name: Optional[str] = None, as_columns=False, include_hidden: bool = True
    ) -> Union[list[str], list[Column]]:
        """
        Returns DB columns as strings or Column objects with proper types
        Optionally, it can filter results by specific object, returning only his signals
        """
        signals = [
            DEFAULT_DELIMITER.join(path)
            if not as_columns
            else Column(DEFAULT_DELIMITER.join(path), python_to_sql(_type))
            for path, _type, has_subtree, _ in self.get_flat_tree(
                include_hidden=include_hidden
            )
            if not has_subtree
        ]

        if name:
            if "." in name:
                name = name.replace(".", "__")

            signals = [
                s
                for s in signals
                if str(s) == name or str(s).startswith(f"{name}{DEFAULT_DELIMITER}")
            ]

        return signals  # type: ignore[return-value]

    def resolve(self, *names: str) -> "SignalSchema":
        schema = {}
        for field in names:
            if not isinstance(field, str):
                raise SignalResolvingTypeError("select()", field)
            schema[field] = self._find_in_tree(field.split("."))

        return SignalSchema(schema)

    def _find_in_tree(self, path: list[str]) -> DataType:
        if val := self.tree.get(".".join(path)):
            # If the path is a single string, we can directly access it
            # without traversing the tree.
            return val[0]

        curr_tree = self.tree
        curr_type = None
        i = 0
        while curr_tree is not None and i < len(path):
            if val := curr_tree.get(path[i]):
                curr_type, curr_tree = val
            else:
                curr_type = None
                break
            i += 1

        if curr_type is None or i < len(path):
            # If we reached the end of the path and didn't find a type,
            # or if we didn't traverse the entire path, raise an error.
            raise SignalResolvingError(path, "is not found")

        return curr_type

    def group_by(
        self, partition_by: Sequence[str], new_column: Sequence[Column]
    ) -> "SignalSchema":
        orig_schema = SignalSchema(copy.deepcopy(self.values))
        schema = orig_schema.to_partial(*partition_by)

        vals = {c.name: sql_to_python(c) for c in new_column}
        return SignalSchema(schema.values | vals)

    def select_except_signals(self, *args: str) -> "SignalSchema":
        def has_signal(signal: str):
            signal = signal.replace(".", DEFAULT_DELIMITER)
            return any(signal == s for s in self.db_signals())

        schema = copy.deepcopy(self.values)
        for signal in args:
            if not isinstance(signal, str):
                raise SignalResolvingTypeError("select_except()", signal)

            if signal not in self.values:
                if has_signal(signal):
                    raise SignalRemoveError(
                        signal.split("."),
                        "select_except() error - removing nested signal would"
                        " break parent schema, which isn't supported.",
                    )
                raise SignalResolvingError(
                    signal.split("."),
                    "select_except() error - the signal does not exist",
                )
            del schema[signal]

        return SignalSchema(schema)

    def clone_without_file_signals(self) -> "SignalSchema":
        schema = copy.deepcopy(self.values)

        for signal in File._datachain_column_types:
            if signal in schema:
                del schema[signal]
        return SignalSchema(schema)

    def mutate(self, args_map: dict) -> "SignalSchema":
        new_values = self.values.copy()
        primitives = (bool, str, int, float)

        for name, value in args_map.items():
            if isinstance(value, Column) and value.name in self.values:
                # renaming existing signal
                del new_values[value.name]
                new_values[name] = self.values[value.name]
                continue
            if isinstance(value, Column):
                # adding new signal from existing signal field
                try:
                    new_values[name] = self.get_column_type(
                        value.name, with_subtree=True
                    )
                    continue
                except SignalResolvingError:
                    pass
            if isinstance(value, Func):
                # adding new signal with function
                new_values[name] = value.get_result_type(self)
                continue
            if isinstance(value, primitives):
                # For primitives, store the type, not the value
                val = literal(value)
                val.type = python_to_sql(type(value))()
                new_values[name] = sql_to_python(val)
                continue
            if isinstance(value, ColumnElement):
                # adding new signal
                new_values[name] = sql_to_python(value)
                continue
            new_values[name] = value

        return SignalSchema(new_values)

    def clone_without_sys_signals(self) -> "SignalSchema":
        schema = copy.deepcopy(self.values)
        schema.pop("sys", None)
        return SignalSchema(schema)

    def merge(
        self,
        right_schema: "SignalSchema",
        rname: str,
    ) -> "SignalSchema":
        schema_right = {
            rname + key if key in self.values else key: type_
            for key, type_ in right_schema.values.items()
        }

        return SignalSchema(self.values | schema_right)

    def append(self, right: "SignalSchema") -> "SignalSchema":
        missing_schema = {
            key: right.values[key]
            for key in [k for k in right.values if k not in self.values]
        }
        return SignalSchema(self.values | missing_schema)

    def get_signals(self, target_type: type[DataModel]) -> Iterator[str]:
        for path, type_, has_subtree, _ in self.get_flat_tree():
            if has_subtree and issubclass(type_, target_type):
                yield ".".join(path)

    def create_model(self, name: str) -> type[DataModel]:
        fields = {key: (value, None) for key, value in self.values.items()}

        return create_model(
            name,
            __base__=(DataModel,),  # type: ignore[call-overload]
            **fields,
        )

    @staticmethod
    def _build_tree(
        values: dict[str, DataType],
    ) -> dict[str, tuple[DataType, Optional[dict]]]:
        return {
            name: (val, SignalSchema._build_tree_for_type(val))
            for name, val in values.items()
        }

    def get_flat_tree(
        self, include_hidden: bool = True
    ) -> Iterator[tuple[list[str], DataType, bool, int]]:
        yield from self._get_flat_tree(self.tree, [], 0, include_hidden)

    def _get_flat_tree(
        self, tree: dict, prefix: list[str], depth: int, include_hidden: bool
    ) -> Iterator[tuple[list[str], DataType, bool, int]]:
        for name, (type_, substree) in tree.items():
            suffix = name.split(".")
            new_prefix = prefix + suffix
            hidden_fields = getattr(type_, "_hidden_fields", None)
            if hidden_fields and substree and not include_hidden:
                substree = {
                    field: info
                    for field, info in substree.items()
                    if field not in hidden_fields
                }

            has_subtree = substree is not None
            yield new_prefix, type_, has_subtree, depth
            if substree is not None:
                yield from self._get_flat_tree(
                    substree, new_prefix, depth + 1, include_hidden
                )

    def print_tree(self, indent: int = 2, start_at: int = 0, file: Optional[IO] = None):
        for path, type_, _, depth in self.get_flat_tree():
            total_indent = start_at + depth * indent
            col_name = " " * total_indent + path[-1]
            col_type = SignalSchema._type_to_str(type_)
            print(col_name, col_type, sep=": ", file=file)

            if get_origin(type_) is list:
                args = get_args(type_)
                if len(args) > 0 and ModelStore.is_pydantic(args[0]):
                    sub_schema = SignalSchema({"* list of": args[0]})
                    sub_schema.print_tree(
                        indent=indent, start_at=total_indent + indent, file=file
                    )

    def get_headers_with_length(self, include_hidden: bool = True):
        paths = [
            path
            for path, _, has_subtree, _ in self.get_flat_tree(
                include_hidden=include_hidden
            )
            if not has_subtree
        ]
        max_length = max([len(path) for path in paths], default=0)
        return [
            path + [""] * (max_length - len(path)) if len(path) < max_length else path
            for path in paths
        ], max_length

    def __or__(self, other):
        return self.__class__(self.values | other.values)

    def __contains__(self, name: str):
        return name in self.values

    def remove(self, name: str):
        return self.values.pop(name)

    @staticmethod
    def _type_to_str(type_: Optional[type], subtypes: Optional[list] = None) -> str:  # noqa: PLR0911
        """Convert a type to a string-based representation."""
        if type_ is None:
            return "NoneType"

        origin = get_origin(type_)

        if origin == Union:
            args = get_args(type_)
            formatted_types = ", ".join(
                SignalSchema._type_to_str(arg, subtypes) for arg in args
            )
            return f"Union[{formatted_types}]"
        if origin == Optional:
            args = get_args(type_)
            type_str = SignalSchema._type_to_str(args[0], subtypes)
            return f"Optional[{type_str}]"
        if origin in (list, List):  # noqa: UP006
            args = get_args(type_)
            type_str = SignalSchema._type_to_str(args[0], subtypes)
            return f"list[{type_str}]"
        if origin in (dict, Dict):  # noqa: UP006
            args = get_args(type_)
            type_str = (
                SignalSchema._type_to_str(args[0], subtypes) if len(args) > 0 else ""
            )
            vals = (
                f", {SignalSchema._type_to_str(args[1], subtypes)}"
                if len(args) > 1
                else ""
            )
            return f"dict[{type_str}{vals}]"
        if origin == Annotated:
            args = get_args(type_)
            return SignalSchema._type_to_str(args[0], subtypes)
        if origin in (Literal, LiteralEx) or type_ in (Literal, LiteralEx):
            return "Literal"
        if Any in (origin, type_):
            return "Any"
        if Final in (origin, type_):
            return "Final"
        if subtypes is not None:
            # Include this type in the list of all subtypes, if requested.
            subtypes.append(type_)
        if not hasattr(type_, "__name__"):
            # This can happen for some third-party or custom types, mostly on Python 3.9
            warnings.warn(
                f"Unable to determine name of type '{type_}'.",
                SignalSchemaWarning,
                stacklevel=2,
            )
            return "Any"
        if ModelStore.is_pydantic(type_):
            ModelStore.register(type_)
            return ModelStore.get_name(type_)
        return type_.__name__

    @staticmethod
    def _build_tree_for_type(
        model: DataType,
    ) -> Optional[dict[str, tuple[DataType, Optional[dict]]]]:
        if (fr := ModelStore.to_pydantic(model)) is not None:
            return SignalSchema._build_tree_for_model(fr)
        return None

    @staticmethod
    def _build_tree_for_model(
        model: type[BaseModel],
    ) -> Optional[dict[str, tuple[DataType, Optional[dict]]]]:
        res: dict[str, tuple[DataType, Optional[dict]]] = {}

        for name, f_info in model.model_fields.items():
            anno = f_info.annotation
            if (fr := ModelStore.to_pydantic(anno)) is not None:
                subtree = SignalSchema._build_tree_for_model(fr)
            else:
                subtree = None
            res[name] = (anno, subtree)  # type: ignore[assignment]

        return res

    def to_partial(self, *columns: str) -> "SignalSchema":  # noqa: C901
        """
        Convert the schema to a partial schema with only the specified columns.

        E.g. if original schema is:

            ```
            signal: Foo@v1
                name: str
                value: float
            count: int
            ```

        Then `to_partial("signal.name", "count")` will return a partial schema:

            ```
            signal: FooPartial@v1
                name: str
            count: int
            ```

        Note that partial schema will have a different name for the custom types
        (e.g. `FooPartial@v1` instead of `Foo@v1`) to avoid conflicts
        with the original schema.

        Args:
            *columns (str): The columns to include in the partial schema.

        Returns:
            SignalSchema: The new partial schema.
        """
        serialized = self.serialize()
        custom_types = serialized.get("_custom_types", {})

        schema: dict[str, Any] = {}
        schema_custom_types: dict[str, CustomType] = {}

        data_model_bases: Optional[list[tuple[str, str, Optional[str]]]] = None

        signal_partials: dict[str, str] = {}
        partial_versions: dict[str, int] = {}

        def _type_name_to_partial(signal_name: str, type_name: str) -> str:
            # Check if we need to create a partial for this type
            # Only create partials for custom types that are in the custom_types dict
            if type_name not in custom_types:
                return type_name

            if "@" in type_name:
                model_name, _ = ModelStore.parse_name_version(type_name)
            else:
                model_name = type_name

            if signal_name not in signal_partials:
                partial_versions.setdefault(model_name, 0)
                partial_versions[model_name] += 1
                version = partial_versions[model_name]
                signal_partials[signal_name] = f"{model_name}Partial{version}"

            return signal_partials[signal_name]

        for column in columns:
            parent_type, parent_type_partial = "", ""
            column_parts = column.split(".")
            for i, signal in enumerate(column_parts):
                if i == 0:
                    if signal not in serialized:
                        raise SignalSchemaError(
                            f"Column {column} not found in the schema"
                        )

                    parent_type = serialized[signal]
                    parent_type_partial = _type_name_to_partial(signal, parent_type)

                    schema[signal] = parent_type_partial

                    # If this is a complex signal without field specifier (just "file")
                    # and it's a custom type, include the entire complex signal
                    if len(column_parts) == 1 and parent_type in custom_types:
                        # Include the entire complex signal - no need to create partial
                        schema[signal] = parent_type
                        continue

                    continue

                if parent_type not in custom_types:
                    raise SignalSchemaError(
                        f"Custom type {parent_type} not found in the schema"
                    )

                custom_type = custom_types[parent_type]
                signal_type = custom_type["fields"].get(signal)
                if not signal_type:
                    raise SignalSchemaError(
                        f"Field {signal} not found in custom type {parent_type}"
                    )

                # Check if this is the last part and if the column type is a complex
                is_last_part = i == len(column_parts) - 1
                is_complex_signal = signal_type in custom_types

                if is_last_part and is_complex_signal:
                    schema[column] = signal_type
                    # Also need to remove the partial schema entry we created for the
                    # parent since we're promoting the nested complex column to root
                    parent_signal = column_parts[0]
                    schema.pop(parent_signal, None)
                    # Don't create partial types for this case
                    break

                # Create partial type for this field
                partial_type = _type_name_to_partial(
                    ".".join(column_parts[: i + 1]),
                    signal_type,
                )

                if parent_type_partial in schema_custom_types:
                    schema_custom_types[parent_type_partial].fields[signal] = (
                        partial_type
                    )
                else:
                    if data_model_bases is None:
                        data_model_bases = SignalSchema._get_bases(DataModel)

                    partial_type_name, _ = ModelStore.parse_name_version(partial_type)
                    schema_custom_types[parent_type_partial] = CustomType(
                        schema_version=2,
                        name=partial_type_name,
                        fields={signal: partial_type},
                        bases=[
                            (partial_type_name, "__main__", partial_type),
                            *data_model_bases,
                        ],
                    )

                parent_type, parent_type_partial = signal_type, partial_type

        if schema_custom_types:
            schema["_custom_types"] = {
                type_name: ct.model_dump()
                for type_name, ct in schema_custom_types.items()
            }

        return SignalSchema.deserialize(schema)
