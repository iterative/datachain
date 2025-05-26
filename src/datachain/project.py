import builtins
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Optional, TypeVar

from datachain.namespace import Namespace

P = TypeVar("P", bound="Project")


@dataclass(frozen=True)
class Project:
    id: int
    uuid: str
    name: str
    description: Optional[str]
    created_at: datetime
    namespace: Namespace

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
        namespace_id: int,
        namespace_uuid: str,
        namespace_name: str,
        namespace_description: Optional[str],
        namespace_created_at: datetime,
        project_id: int,
        uuid: str,
        name: str,
        description: Optional[str],
        created_at: datetime,
        project_namespace_id: int,
    ) -> "Project":
        namespace = Namespace.parse(
            namespace_id,
            namespace_uuid,
            namespace_name,
            namespace_description,
            namespace_created_at,
        )

        return cls(project_id, uuid, name, description, created_at, namespace)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        namespace = Namespace.from_dict(d.pop("namespace"))
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs, namespace=namespace)
