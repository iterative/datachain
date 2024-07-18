import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TypeVar

J = TypeVar("J", bound="Job")


@dataclass
class Job:
    id: str
    name: str
    status: int
    created_at: datetime
    query: str
    query_type: int
    workers: int
    params: dict[str, str]
    metrics: dict[str, Any]
    finished_at: Optional[datetime] = None
    python_version: Optional[str] = None
    error_message: str = ""
    error_stack: str = ""

    @classmethod
    def parse(
        cls: type[J],
        id: str,
        name: str,
        status: int,
        created_at: datetime,
        finished_at: Optional[datetime],
        query: str,
        query_type: int,
        workers: int,
        python_version: Optional[str],
        error_message: str,
        error_stack: str,
        params: str,
        metrics: str,
    ) -> "Job":
        return cls(
            id,
            name,
            status,
            created_at,
            query,
            query_type,
            workers,
            json.loads(params),
            json.loads(metrics),
            finished_at,
            python_version,
            error_message,
            error_stack,
        )
