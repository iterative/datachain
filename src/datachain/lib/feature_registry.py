import inspect
import logging
from typing import Any, ClassVar, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Registry:
    reg: ClassVar[dict[str, dict[int, Any]]] = {}

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
    def add(cls, fr: type) -> None:
        name = fr.__name__
        if name not in cls.reg:
            cls.reg[name] = {}
        version = Registry.get_version(fr)
        if version in cls.reg[name]:
            full_name = f"{name}@{version}"
            logger.warning("Feature %s is already registered", full_name)
        cls.reg[name][version] = fr

        if issubclass(fr, BaseModel):
            for f_info in fr.model_fields.values():
                anno = f_info.annotation
                if inspect.isclass(anno) and issubclass(anno, BaseModel):
                    cls.add(anno)

    @classmethod
    def get(cls, name: str, version: Optional[int] = None) -> Optional[type]:
        class_dict = cls.reg.get(name, None)
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
        if fr.__name__ in cls.reg and version in cls.reg[fr.__name__]:
            del cls.reg[fr.__name__][version]
