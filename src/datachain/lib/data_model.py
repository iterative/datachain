from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Union

from pydantic import BaseModel

from datachain.lib.feature import FeatureType, is_feature
from datachain.lib.feature_registry import Registry

if TYPE_CHECKING:
    from datachain.catalog import Catalog


class DataModel(BaseModel):
    _version: ClassVar[int] = 1

    def get_value(self):
        return None

    # It automatically registers every declared DataModel child class
    @classmethod
    def __pydantic_init_subclass__(cls):
        Registry.add(cls)

    @staticmethod
    def register(models: Union[FeatureType, Sequence[FeatureType]]):
        if not isinstance(models, Sequence):
            models = [models]
        for val in models:
            if is_feature(val):
                Registry.add(val)


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
