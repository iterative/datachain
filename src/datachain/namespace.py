import builtins
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar

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

    @staticmethod
    def allowed_to_create() -> bool:
        """
        User cannot create it's own custom namespace explicitly by default.
        """
        return False

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
