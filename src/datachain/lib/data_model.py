from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Union, get_args, get_origin

from pydantic import BaseModel

from datachain.lib.model_store import ModelStore

if TYPE_CHECKING:
    from datachain.catalog import Catalog

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


class DataModel(BaseModel):
    _version: ClassVar[int] = 1

    def get_value(self):
        """Getting value from data. It's used in conjunction with method that operate
        with raw data such as to_pytorch(). In contrast to method that operated with
        data structures such as pydantic"""
        return

    @classmethod
    def __pydantic_init_subclass__(cls):
        """It automatically registers every declared DataModel child class."""
        ModelStore.add(cls)

    @staticmethod
    def register(models: Union[DataType, Sequence[DataType]]):
        """For registering classes manually. It accepts a single class or a sequence of
        classes."""
        if not isinstance(models, Sequence):
            models = [models]
        for val in models:
            ModelStore.add(val)


class FileBasic(DataModel):
    def _set_stream(self, catalog: "Catalog", caching_enabled: bool = False) -> None:
        pass

    def open(self):
        raise NotImplementedError

    def read(self):
        with self.open() as stream:
            return stream.read()

    def get_value(self):
        return self.read()


def is_chain_type(t: type) -> bool:
    if ModelStore.is_pydantic(t):
        return True
    if any(t is ft or t is get_args(ft)[0] for ft in get_args(StandardType)):
        return True

    if get_origin(t) is list and len(get_args(t)) == 1:
        return is_chain_type(get_args(t)[0])

    return False
