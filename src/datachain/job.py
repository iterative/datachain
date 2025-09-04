import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TypeVar, Union

J = TypeVar("J", bound="Job")
JQS = TypeVar("JQS", bound="JobQueryStep")


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
        cls,
        id: Union[str, uuid.UUID],
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
            str(id),
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


@dataclass
class JobQueryStep:
    """
    Class that represents one chain that results in a saved dataset inside a job
    query script. One job can have multiple chains and produced datasets so it can
    have multiple query steps as well.
    """

    id: str
    job_id: str
    status: int
    started_at: datetime
    finished_at: Optional[datetime] = None
    error_message: str = ""
    error_stack: str = ""

    @classmethod
    def parse(
        cls,
        id: Union[str, uuid.UUID],
        job_id: str,
        status: int,
        error_message: str,
        error_stack: str,
        started_at: datetime,
        finished_at: Optional[datetime],
    ) -> "JobQueryStep":
        return cls(
            str(id),
            job_id,
            status,
            started_at,
            finished_at,
            error_message,
            error_stack,
        )
