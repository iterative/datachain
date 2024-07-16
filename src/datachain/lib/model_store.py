import logging
from typing import ClassVar, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelStore:
    store: ClassVar[dict[str, dict[int, type[BaseModel]]]] = {}

    @classmethod
    def get_version(cls, model: type[BaseModel]) -> int:
        if not hasattr(model, "_version"):
            return 0
        return model._version

    @classmethod
    def get_name(cls, model) -> str:
        if (version := cls.get_version(model)) > 0:
            return f"{model.__name__}@v{version}"
        return model.__name__

    @classmethod
    def add(cls, fr: type):
        if (model := ModelStore.to_pydantic(fr)) is None:
            return

        name = model.__name__
        if name not in cls.store:
            cls.store[name] = {}
        version = ModelStore.get_version(model)
        cls.store[name][version] = model

        for f_info in model.model_fields.values():
            if (anno := ModelStore.to_pydantic(f_info.annotation)) is not None:
                cls.add(anno)

    @classmethod
    def get(cls, name: str, version: Optional[int] = None) -> Optional[type]:
        class_dict = cls.store.get(name, None)
        if class_dict is None:
            return None
        if version is None:
            max_ver = max(class_dict.keys(), default=None)
            if max_ver is None:
                return None
            return class_dict[max_ver]
        return class_dict.get(version, None)

    @classmethod
    def parse_name_version(cls, fullname: str) -> tuple[str, int]:
        name = fullname
        version = 0

        if "@" in fullname:
            name, version_str = fullname.split("@")
            if version_str.strip() != "":
                version = int(version_str[1:])

        return name, version

    @classmethod
    def remove(cls, fr: type) -> None:
        version = fr._version  # type: ignore[attr-defined]
        if fr.__name__ in cls.store and version in cls.store[fr.__name__]:
            del cls.store[fr.__name__][version]

    @staticmethod
    def is_pydantic(val):
        return not hasattr(val, "__origin__") and issubclass(val, BaseModel)

    @staticmethod
    def to_pydantic(val) -> Optional[type[BaseModel]]:
        if val is None or not ModelStore.is_pydantic(val):
            return None
        return val
