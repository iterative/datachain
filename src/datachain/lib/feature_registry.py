import logging
from typing import Any, ClassVar, Optional

logger = logging.getLogger(__name__)


class Registry:
    reg: ClassVar[dict[str, dict[int, Any]]] = {}

    @classmethod
    def add(cls, fr: type) -> None:
        if fr._is_hidden():  # type: ignore[attr-defined]
            return
        name = fr.__name__
        if name not in cls.reg:
            cls.reg[name] = {}
        version = fr._version  # type: ignore[attr-defined]
        if version in cls.reg[name]:
            full_name = f"{name}@{version}"
            logger.warning("Feature %s is already registered", full_name)
        cls.reg[name][version] = fr

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
        version = 1

        if "@" in fullname:
            name, version_str = fullname.split("@")
            if version_str.strip() != "":
                version = int(version_str)

        return name, version

    @classmethod
    def remove(cls, fr: type) -> None:
        version = fr._version  # type: ignore[attr-defined]
        if fr.__name__ in cls.reg and version in cls.reg[fr.__name__]:
            del cls.reg[fr.__name__][version]
