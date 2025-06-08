import builtins
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Optional, TypeVar

N = TypeVar("N", bound="Namespace")


@dataclass(frozen=True)
class Namespace:
    id: int
    uuid: str
    name: str
    description: Optional[str]
    created_at: datetime

    @staticmethod
    def default() -> str:
        """Name of default namespace"""
        return "local"

    @staticmethod
    def reserved_names() -> list[str]:
        """what names cannot be used when creating a namespace"""
        return [Namespace.default()]

    @classmethod
    def parse(
        cls: builtins.type[N],
        id: int,
        uuid: str,
        name: str,
        description: Optional[str],
        created_at: datetime,
    ) -> "Namespace":
        return cls(id, uuid, name, description, created_at)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Namespace":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs)
