from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Optional, Union, get_args, get_origin

from pydantic import AliasChoices, BaseModel, Field, create_model

from datachain.lib.model_store import ModelStore
from datachain.lib.utils import normalize_col_names

StandardType = Union[
    type[int],
    type[str],
    type[float],
    type[bool],
    type[list],
    type[dict],
    type[bytes],
    type[datetime],
]
DataType = Union[type[BaseModel], StandardType]
DataTypeNames = "BaseModel, int, str, float, bool, list, dict, bytes, datetime"
DataValue = Union[BaseModel, int, str, float, bool, list, dict, bytes, datetime]


class DataModel(BaseModel):
    """Pydantic model wrapper that registers model with `DataChain`."""

    _version: ClassVar[int] = 1
    _hidden_fields: ClassVar[list[str]] = []

    @classmethod
    def __pydantic_init_subclass__(cls):
        """It automatically registers every declared DataModel child class."""
        ModelStore.register(cls)

    @staticmethod
    def register(models: Union[DataType, Sequence[DataType]]):
        """For registering classes manually. It accepts a single class or a sequence of
        classes."""
        if not isinstance(models, Sequence):
            models = [models]
        for val in models:
            ModelStore.register(val)

    @classmethod
    def hidden_fields(cls) -> list[str]:
        """Returns a list of fields that should be hidden from the user."""
        return cls._hidden_fields


def is_chain_type(t: type) -> bool:
    """Return true if type is supported by `DataChain`."""
    if ModelStore.is_pydantic(t):
        return True
    if any(t is ft or t is get_args(ft)[0] for ft in get_args(StandardType)):
        return True

    orig = get_origin(t)
    args = get_args(t)
    if orig is list and len(args) == 1:
        return is_chain_type(get_args(t)[0])

    if orig is Union and len(args) == 2 and (type(None) in args):
        return is_chain_type(args[0])

    return False


def dict_to_data_model(
    name: str,
    data_dict: dict[str, DataType],
    original_names: Optional[list[str]] = None,
) -> type[BaseModel]:
    if not original_names:
        # Gets a map of a normalized_name -> original_name
        columns = normalize_col_names(list(data_dict))
        data_dict = dict(zip(columns.keys(), data_dict.values()))
        original_names = list(columns.values())

    fields = {
        name: (
            anno,
            Field(
                validation_alias=AliasChoices(name, original_names[idx] or name),
                default=None,
            ),
        )
        for idx, (name, anno) in enumerate(data_dict.items())
    }

    class _DataModelStrict(BaseModel, extra="forbid"):
        pass

    return create_model(
        name,
        __base__=_DataModelStrict,
        **fields,
    )  # type: ignore[call-overload]
