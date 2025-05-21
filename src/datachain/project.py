import builtins
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar

P = TypeVar("P", bound="Project")


@dataclass
class Project:
    id: int
    uuid: str
    name: str
    description: Optional[str]
    created_at: datetime
    namespace_id: int

    @staticmethod
    def default() -> str:
        """Name of default project"""
        return "local"

    @staticmethod
    def reserved_names() -> list[str]:
        """what names cannot be used when creating a project"""
        return [Project.default()]

    @staticmethod
    def allowed_to_create() -> bool:
        """
        User cannot create it's own custom project explicitly by default.
        """
        return False

    @classmethod
    def parse(
        cls: builtins.type[P],
        id: int,
        uuid: str,
        name: str,
        description: Optional[str],
        created_at: datetime,
        namespace_id: int,
    ) -> "Project":
        return cls(id, uuid, name, description, created_at, namespace_id)
