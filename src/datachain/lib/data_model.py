from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar, Union, get_args, get_origin

from pydantic import BaseModel, create_model

from datachain.lib.model_store import ModelStore

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
DataValuesType = Union[BaseModel, int, str, float, bool, list, dict, bytes, datetime]


class DataModel(BaseModel):
    """Pydantic model wrapper that registers model with `DataChain`."""

    _version: ClassVar[int] = 1

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


def dict_to_data_model(name: str, data_dict: dict[str, DataType]) -> type[BaseModel]:
    fields = {name: (anno, ...) for name, anno in data_dict.items()}
    return create_model(
        name,
        __base__=(DataModel,),  # type: ignore[call-overload]
        **fields,
    )  # type: ignore[call-overload]
