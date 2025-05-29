import builtins
import json
from dataclasses import dataclass, fields
from datetime import datetime
from functools import cached_property
from typing import (
    Any,
    NewType,
    Optional,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

from datachain import semver
from datachain.error import DatasetVersionNotFoundError
from datachain.sql.types import NAME_TYPES_MAPPING, SQLType

T = TypeVar("T", bound="DatasetRecord")
LT = TypeVar("LT", bound="DatasetListRecord")
V = TypeVar("V", bound="DatasetVersion")
LV = TypeVar("LV", bound="DatasetListVersion")
DD = TypeVar("DD", bound="DatasetDependency")

DATASET_PREFIX = "ds://"
QUERY_DATASET_PREFIX = "ds_query_"
LISTING_PREFIX = "lst__"

DEFAULT_DATASET_VERSION = "1.0.0"


# StorageURI represents a normalised URI to a valid storage location (full bucket or
# absolute local path).
# Valid examples: s3://foo, file:///var/data
# Invalid examples: s3://foo/, s3://foo/bar, file://~
StorageURI = NewType("StorageURI", str)


def parse_dataset_uri(uri: str) -> tuple[str, Optional[str]]:
    """
    Parse dataser uri to extract name and version out of it (if version is defined)
    Example:
        Input: ds://zalando@v3.0.1
        Output: (zalando, 3.0.1)
    """
    p = urlparse(uri)
    if p.scheme != "ds":
        raise Exception("Dataset uri should start with ds://")
    s = p.netloc.split("@v")
    name = s[0]
    if len(s) == 1:
        return name, None
    if len(s) != 2:
        raise Exception(
            "Wrong dataset uri format, it should be: ds://<name>@v<version>"
        )
    return name, s[1]


def create_dataset_uri(name: str, version: Optional[str] = None) -> str:
    """
    Creates a dataset uri based on dataset name and optionally version
    Example:
        Input: zalando, 3.0.1
        Output: ds//zalando@v3.0.1
    """
    uri = f"{DATASET_PREFIX}{name}"
    if version:
        uri += f"@v{version}"

    return uri


class DatasetDependencyType:
    DATASET = "dataset"
    STORAGE = "storage"


@dataclass
class DatasetDependency:
    id: int
    type: str
    name: str
    version: str
    created_at: datetime
    dependencies: list[Optional["DatasetDependency"]]

    @property
    def dataset_name(self) -> str:
        """Returns clean dependency dataset name"""
        from datachain.lib.listing import parse_listing_uri

        if self.type == DatasetDependencyType.DATASET:
            return self.name

        list_dataset_name, _, _ = parse_listing_uri(self.name.strip("/"))
        assert list_dataset_name
        return list_dataset_name

    @classmethod
    def parse(
        cls: builtins.type[DD],
        id: int,
        dataset_id: Optional[int],
        dataset_version_id: Optional[int],
        dataset_name: Optional[str],
        dataset_version: Optional[str],
        dataset_version_created_at: Optional[datetime],
    ) -> Optional["DatasetDependency"]:
        from datachain.lib.listing import is_listing_dataset

        if not dataset_id:
            return None

        assert dataset_name is not None

        return cls(
            id,
            (
                DatasetDependencyType.STORAGE
                if is_listing_dataset(dataset_name)
                else DatasetDependencyType.DATASET
            ),
            dataset_name,
            (
                dataset_version  # type: ignore[arg-type]
                if dataset_version
                else None
            ),
            dataset_version_created_at,  # type: ignore[arg-type]
            [],
        )

    @property
    def is_dataset(self) -> bool:
        return self.type == DatasetDependencyType.DATASET

    def __eq__(self, other):
        if not isinstance(other, DatasetDependency):
            return False

        return (
            self.type == other.type
            and self.name == other.name
            and self.version == other.version
        )

    def __hash__(self):
        return hash(f"{self.type}_{self.name}_{self.version}")


class DatasetStatus:
    CREATED = 1
    PENDING = 2
    FAILED = 3
    COMPLETE = 4
    STALE = 6


@dataclass
class DatasetVersion:
    id: int
    uuid: str
    dataset_id: int
    version: str
    status: int
    feature_schema: dict
    created_at: datetime
    finished_at: Optional[datetime]
    error_message: str
    error_stack: str
    script_output: str
    schema: dict[str, Union[SQLType, type[SQLType]]]
    num_objects: Optional[int]
    size: Optional[int]
    _preview_data: Optional[Union[str, list[dict]]]
    sources: str = ""
    query_script: str = ""
    job_id: Optional[str] = None

    @classmethod
    def parse(  # noqa: PLR0913
        cls,
        id: int,
        uuid: str,
        dataset_id: int,
        version: str,
        status: int,
        feature_schema: Optional[str],
        created_at: datetime,
        finished_at: Optional[datetime],
        error_message: str,
        error_stack: str,
        script_output: str,
        num_objects: Optional[int],
        size: Optional[int],
        preview: Optional[Union[str, list[dict]]],
        schema: dict[str, Union[SQLType, type[SQLType]]],
        sources: str = "",
        query_script: str = "",
        job_id: Optional[str] = None,
    ):
        return cls(
            id,
            uuid,
            dataset_id,
            version,
            status,
            json.loads(feature_schema) if feature_schema else {},
            created_at,
            finished_at,
            error_message,
            error_stack,
            script_output,
            schema,
            num_objects,
            size,
            preview,
            sources,
            query_script,
            job_id,
        )

    @property
    def version_value(self) -> int:
        return semver.value(self.version)

    def __eq__(self, other):
        if not isinstance(other, DatasetVersion):
            return False
        return self.version == other.version and self.dataset_id == other.dataset_id

    def __lt__(self, other):
        if not isinstance(other, DatasetVersion):
            return False
        return self.version_value < other.version_value

    def __hash__(self):
        return hash(f"{self.dataset_id}_{self.version}")

    def is_final_status(self) -> bool:
        return self.status in [
            DatasetStatus.FAILED,
            DatasetStatus.COMPLETE,
            DatasetStatus.STALE,
        ]

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @property
    def serialized_schema(self) -> dict[str, Any]:
        return {
            c_name: c_type.to_dict()
            if isinstance(c_type, SQLType)
            else c_type().to_dict()
            for c_name, c_type in self.schema.items()
        }

    @cached_property
    def preview(self) -> Optional[list[dict]]:
        if isinstance(self._preview_data, str):
            return json.loads(self._preview_data)
        return self._preview_data if self._preview_data else None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetVersion":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        if not hasattr(kwargs, "_preview_data"):
            kwargs["_preview_data"] = d.get("preview")
        return cls(**kwargs)


@dataclass
class DatasetListVersion:
    id: int
    uuid: str
    dataset_id: int
    version: str
    status: int
    created_at: datetime
    finished_at: Optional[datetime]
    error_message: str
    error_stack: str
    num_objects: Optional[int]
    size: Optional[int]
    query_script: str = ""
    job_id: Optional[str] = None

    @classmethod
    def parse(
        cls,
        id: int,
        uuid: str,
        dataset_id: int,
        version: str,
        status: int,
        created_at: datetime,
        finished_at: Optional[datetime],
        error_message: str,
        error_stack: str,
        num_objects: Optional[int],
        size: Optional[int],
        query_script: str = "",
        job_id: Optional[str] = None,
        **kwargs,
    ):
        return cls(
            id,
            uuid,
            dataset_id,
            version,
            status,
            created_at,
            finished_at,
            error_message,
            error_stack,
            num_objects,
            size,
            query_script,
            job_id,
        )

    def __hash__(self):
        return hash(f"{self.dataset_id}_{self.version}")

    @property
    def version_value(self) -> int:
        return semver.value(self.version)


@dataclass
class DatasetRecord:
    id: int
    name: str
    description: Optional[str]
    attrs: list[str]
    schema: dict[str, Union[SQLType, type[SQLType]]]
    feature_schema: dict
    versions: list[DatasetVersion]
    status: int = DatasetStatus.CREATED
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: str = ""
    error_stack: str = ""
    script_output: str = ""
    sources: str = ""
    query_script: str = ""

    @staticmethod
    def parse_schema(
        ct: dict[str, Any],
    ) -> dict[str, Union[SQLType, type[SQLType]]]:
        return {
            c_name: NAME_TYPES_MAPPING[c_type["type"]].from_dict(c_type)  # type: ignore [attr-defined]
            for c_name, c_type in ct.items()
        }

    @classmethod
    def parse(  # noqa: PLR0913
        cls,
        id: int,
        name: str,
        description: Optional[str],
        attrs: str,
        status: int,
        feature_schema: Optional[str],
        created_at: datetime,
        finished_at: Optional[datetime],
        error_message: str,
        error_stack: str,
        script_output: str,
        sources: str,
        query_script: str,
        schema: str,
        version_id: int,
        version_uuid: str,
        version_dataset_id: int,
        version: str,
        version_status: int,
        version_feature_schema: Optional[str],
        version_created_at: datetime,
        version_finished_at: Optional[datetime],
        version_error_message: str,
        version_error_stack: str,
        version_script_output: str,
        version_num_objects: Optional[int],
        version_size: Optional[int],
        version_preview: Optional[str],
        version_sources: Optional[str],
        version_query_script: Optional[str],
        version_schema: str,
        version_job_id: Optional[str] = None,
    ) -> "DatasetRecord":
        attrs_lst: list[str] = json.loads(attrs) if attrs else []
        schema_dct: dict[str, Any] = json.loads(schema) if schema else {}
        version_schema_dct: dict[str, str] = (
            json.loads(version_schema) if version_schema else {}
        )

        dataset_version = DatasetVersion.parse(
            version_id,
            version_uuid,
            version_dataset_id,
            version,
            version_status,
            version_feature_schema,
            version_created_at,
            version_finished_at,
            version_error_message,
            version_error_stack,
            version_script_output,
            version_num_objects,
            version_size,
            version_preview,
            cls.parse_schema(version_schema_dct),  # type: ignore[arg-type]
            version_sources,  # type: ignore[arg-type]
            version_query_script,  # type: ignore[arg-type]
            version_job_id,
        )

        return cls(
            id,
            name,
            description,
            attrs_lst,
            cls.parse_schema(schema_dct),  # type: ignore[arg-type]
            json.loads(feature_schema) if feature_schema else {},
            [dataset_version],
            status,
            created_at,
            finished_at,
            error_message,
            error_stack,
            script_output,
            sources,
            query_script,
        )

    @property
    def serialized_schema(self) -> dict[str, Any]:
        return {
            c_name: c_type.to_dict()
            if isinstance(c_type, SQLType)
            else c_type().to_dict()
            for c_name, c_type in self.schema.items()
        }

    def get_schema(self, version: str) -> dict[str, Union[SQLType, type[SQLType]]]:
        return self.get_version(version).schema if version else self.schema

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def merge_versions(self, other: "DatasetRecord") -> "DatasetRecord":
        """Merge versions from another dataset"""
        if other.id != self.id:
            raise RuntimeError("Cannot merge versions of datasets with different ids")
        if not other.versions:
            # nothing to merge
            return self
        if not self.versions:
            self.versions = []

        self.versions = list(set(self.versions + other.versions))
        self.versions.sort(key=lambda v: v.version_value)
        return self

    def has_version(self, version: str) -> bool:
        return version in [v.version for v in self.versions]

    def is_valid_next_version(self, version: str) -> bool:
        """
        Checks if a number can be a valid next latest version for dataset.
        The only rule is that it cannot be lower than current latest version
        """
        return not (
            self.latest_version
            and semver.value(self.latest_version) >= semver.value(version)
        )

    def get_version(self, version: str) -> DatasetVersion:
        if not self.has_version(version):
            raise DatasetVersionNotFoundError(
                f"Dataset {self.name} does not have version {version}"
            )
        return next(
            v
            for v in self.versions  # type: ignore [union-attr]
            if v.version == version
        )

    def get_version_by_uuid(self, uuid: str) -> DatasetVersion:
        try:
            return next(
                v
                for v in self.versions  # type: ignore [union-attr]
                if v.uuid == uuid
            )
        except StopIteration:
            raise DatasetVersionNotFoundError(
                f"Dataset {self.name} does not have version with uuid {uuid}"
            ) from None

    def remove_version(self, version: str) -> None:
        if not self.versions or not self.has_version(version):
            return

        self.versions = [v for v in self.versions if v.version != version]

    def identifier(self, version: str) -> str:
        """
        Get identifier in the form my-dataset@v3.0.1
        """
        if not self.has_version(version):
            raise DatasetVersionNotFoundError(
                f"Dataset {self.name} doesn't have a version {version}"
            )
        return f"{self.name}@v{version}"

    def uri(self, version: str) -> str:
        """
        Dataset uri example: ds://dogs@v3.0.1
        """
        identifier = self.identifier(version)
        return f"{DATASET_PREFIX}{identifier}"

    @property
    def next_version_major(self) -> str:
        """
        Returns the next auto-incremented version if the major part is being bumped.
        """
        if not self.versions:
            return "1.0.0"

        major, minor, patch = semver.parse(self.latest_version)
        return semver.create(major + 1, 0, 0)

    @property
    def next_version_minor(self) -> str:
        """
        Returns the next auto-incremented version if the minor part is being bumped.
        """
        if not self.versions:
            return "1.0.0"

        major, minor, patch = semver.parse(self.latest_version)
        return semver.create(major, minor + 1, 0)

    @property
    def next_version_patch(self) -> str:
        """
        Returns the next auto-incremented version if the patch part is being bumped.
        """
        if not self.versions:
            return "1.0.0"

        major, minor, patch = semver.parse(self.latest_version)
        return semver.create(major, minor, patch + 1)

    @property
    def latest_version(self) -> str:
        """Returns latest version of a dataset"""
        return max(self.versions).version

    def latest_major_version(self, major: int) -> Optional[str]:
        """
        Returns latest specific major version, e.g if dataset has versions:
            - 1.4.1
            - 2.0.1
            - 2.1.1
            - 2.4.0
        and we call `.latest_major_version(2)` it will return: "2.4.0".
        If no major version is find with input value, None will be returned
        """
        versions = [v for v in self.versions if semver.parse(v.version)[0] == major]
        if not versions:
            return None
        return max(versions).version

    @property
    def prev_version(self) -> Optional[str]:
        """Returns previous version of a dataset"""
        if len(self.versions) == 1:
            return None

        return sorted(self.versions)[-2].version

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetRecord":
        versions = [DatasetVersion.from_dict(v) for v in d.pop("versions", [])]
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs, versions=versions)


@dataclass
class DatasetListRecord:
    id: int
    name: str
    description: Optional[str]
    attrs: list[str]
    versions: list[DatasetListVersion]
    created_at: Optional[datetime] = None

    @classmethod
    def parse(  # noqa: PLR0913
        cls,
        id: int,
        name: str,
        description: Optional[str],
        attrs: str,
        created_at: datetime,
        version_id: int,
        version_uuid: str,
        version_dataset_id: int,
        version: str,
        version_status: int,
        version_created_at: datetime,
        version_finished_at: Optional[datetime],
        version_error_message: str,
        version_error_stack: str,
        version_num_objects: Optional[int],
        version_size: Optional[int],
        version_query_script: Optional[str],
        version_job_id: Optional[str] = None,
    ) -> "DatasetListRecord":
        attrs_lst: list[str] = json.loads(attrs) if attrs else []

        dataset_version = DatasetListVersion.parse(
            version_id,
            version_uuid,
            version_dataset_id,
            version,
            version_status,
            version_created_at,
            version_finished_at,
            version_error_message,
            version_error_stack,
            version_num_objects,
            version_size,
            version_query_script,  # type: ignore[arg-type]
            version_job_id,
        )

        return cls(
            id,
            name,
            description,
            attrs_lst,
            [dataset_version],
            created_at,
        )

    def merge_versions(self, other: "DatasetListRecord") -> "DatasetListRecord":
        """Merge versions from another dataset"""
        if other.id != self.id:
            raise RuntimeError("Cannot merge versions of datasets with different ids")
        if not other.versions:
            # nothing to merge
            return self
        if not self.versions:
            self.versions = []

        self.versions = list(set(self.versions + other.versions))
        self.versions.sort(key=lambda v: v.version_value)
        return self

    def latest_version(self) -> DatasetListVersion:
        return max(self.versions, key=lambda v: v.version_value)

    @property
    def is_bucket_listing(self) -> bool:
        """
        For bucket listing we implicitly create underlying dataset to hold data. This
        method is checking if this is one of those datasets.
        """
        from datachain.client import Client

        # TODO refactor and maybe remove method in
        # https://github.com/iterative/datachain/issues/318
        return Client.is_data_source_uri(self.name) or self.name.startswith(
            LISTING_PREFIX
        )

    def has_version_with_uuid(self, uuid: str) -> bool:
        return any(v.uuid == uuid for v in self.versions)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetListRecord":
        versions = [DatasetListVersion.parse(**v) for v in d.get("versions", [])]
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kwargs["versions"] = versions
        return cls(**kwargs)


class RowDict(dict):
    pass
