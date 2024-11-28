import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

from pydantic import Field, field_validator

from datachain.dataset import (
    DatasetListRecord,
    DatasetListVersion,
    DatasetStatus,
)
from datachain.job import Job
from datachain.lib.data_model import DataModel
from datachain.utils import TIME_ZERO

if TYPE_CHECKING:
    from typing_extensions import Self


class DatasetInfo(DataModel):
    name: str
    uuid: str = Field(default=str(uuid4()))
    version: int = Field(default=1)
    status: int = Field(default=DatasetStatus.CREATED)
    created_at: datetime = Field(default=TIME_ZERO)
    finished_at: Optional[datetime] = Field(default=None)
    num_objects: Optional[int] = Field(default=None)
    size: Optional[int] = Field(default=None)
    params: dict[str, str] = Field(default={})
    metrics: dict[str, Any] = Field(default={})
    error_message: str = Field(default="")
    error_stack: str = Field(default="")

    @staticmethod
    def _validate_dict(
        v: Optional[Union[str, dict]],
    ) -> dict:
        if v is None or v == "":
            return {}
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"Unable to convert string '{v}' to dict for Dataset feature: {e}"
                ) from None
        return v

    # Workaround for empty JSONs converted to empty strings in some DBs.
    @field_validator("params", mode="before")
    @classmethod
    def validate_location(cls, v):
        return cls._validate_dict(v)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, v):
        return cls._validate_dict(v)

    @classmethod
    def from_models(
        cls,
        dataset: DatasetListRecord,
        version: DatasetListVersion,
        job: Optional[Job],
    ) -> "Self":
        return cls(
            uuid=version.uuid,
            name=dataset.name,
            version=version.version,
            status=version.status,
            created_at=version.created_at,
            finished_at=version.finished_at,
            num_objects=version.num_objects,
            size=version.size,
            params=job.params if job else {},
            metrics=job.metrics if job else {},
            error_message=version.error_message,
            error_stack=version.error_stack,
        )
