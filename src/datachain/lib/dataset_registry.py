import json
from datetime import datetime
from typing import Any, ClassVar, Optional, Union

from pydantic import Field, field_validator

from datachain.dataset import DatasetRecord, DatasetStatus, DatasetVersion
from datachain.job import Job
from datachain.lib.feature import Feature
from datachain.sql.types import JSON, DateTime, Int, String
from datachain.utils import TIME_ZERO


class Dataset(Feature):
    dataset_id: int
    version_id: int
    name: str
    version: int = Field(default=1)
    status: int = Field(default=DatasetStatus.CREATED)
    created_at: datetime = Field(default=TIME_ZERO)
    finished_at: Optional[datetime] = Field(default=None)
    num_objects: Optional[int] = Field(default=None)
    size: Optional[int] = Field(default=None)
    params: dict[str, str] = Field(default=dict)
    metrics: dict[str, Any] = Field(default=dict)

    _datachain_column_types: ClassVar[dict[str, Any]] = {
        "dataset_id": Int,
        "version_id": Int,
        "name": String,
        "version": Int,
        "status": Int,
        "created_at": DateTime,
        "finished_at": DateTime,
        "num_objects": Int,
        "size": Int,
        "params": JSON,
        "metrics": JSON,
    }

    _unique_id_keys: ClassVar[list[str]] = ["name", "version"]

    @staticmethod
    def to_dict(
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
        return Dataset.to_dict(v)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, v):
        return Dataset.to_dict(v)

    @classmethod
    def from_models(
        cls,
        dataset: DatasetRecord,
        version: DatasetVersion,
        job: Optional[Job],
    ) -> "Dataset":
        return cls(
            dataset_id=dataset.id,
            version_id=version.id,
            name=dataset.name,
            version=version.version,
            status=version.status,
            created_at=version.created_at,
            finished_at=version.finished_at,
            num_objects=version.num_objects,
            size=version.size,
            params=job.params if job else {},
            metrics=job.metrics if job else {},
        )
