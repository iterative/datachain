import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime, timezone
from functools import cached_property, reduce
from itertools import groupby
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Table,
    Text,
    UniqueConstraint,
    select,
)

from datachain.data_storage import JobQueryType, JobStatus
from datachain.data_storage.serializer import Serializable
from datachain.dataset import (
    DatasetDependency,
    DatasetListRecord,
    DatasetListVersion,
    DatasetRecord,
    DatasetStatus,
    DatasetVersion,
    StorageURI,
)
from datachain.error import (
    DatasetNotFoundError,
    TableMissingError,
)
from datachain.job import Job
from datachain.utils import JSONSerialize

if TYPE_CHECKING:
    from sqlalchemy import Delete, Insert, Select, Update
    from sqlalchemy.schema import SchemaItem

    from datachain.data_storage import schema
    from datachain.data_storage.db_engine import DatabaseEngine

logger = logging.getLogger("datachain")


class AbstractMetastore(ABC, Serializable):
    """
    Abstract Metastore class.
    This manages the storing, searching, and retrieval of indexed metadata.
    """

    uri: StorageURI

    schema: "schema.Schema"
    dataset_class: type[DatasetRecord] = DatasetRecord
    dataset_list_class: type[DatasetListRecord] = DatasetListRecord
    dataset_list_version_class: type[DatasetListVersion] = DatasetListVersion
    dependency_class: type[DatasetDependency] = DatasetDependency
    job_class: type[Job] = Job

    def __init__(
        self,
        uri: Optional[StorageURI] = None,
    ):
        self.uri = uri or StorageURI("")

    def __enter__(self) -> "AbstractMetastore":
        """Returns self upon entering context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Default behavior is to do nothing, as connections may be shared."""

    @abstractmethod
    def clone(
        self,
        uri: Optional[StorageURI] = None,
        use_new_connection: bool = False,
    ) -> "AbstractMetastore":
        """Clones AbstractMetastore implementation for some Storage input.
        Setting use_new_connection will always use a new database connection.
        New connections should only be used if needed due to errors with
        closed connections."""

    def close(self) -> None:
        """Closes any active database or HTTP connections."""

    def close_on_exit(self) -> None:
        """Closes any active database or HTTP connections, called on Session exit or
        for test cleanup only, as some Metastore implementations may handle this
        differently."""
        self.close()

    def cleanup_tables(self, temp_table_names: list[str]) -> None:
        """Cleanup temp tables."""

    def cleanup_for_tests(self) -> None:
        """Cleanup for tests."""

    #
    # Datasets
    #

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        status: int = DatasetStatus.CREATED,
        sources: Optional[list[str]] = None,
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
    ) -> DatasetRecord:
        """Creates new dataset."""

    @abstractmethod
    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: int,
        status: int,
        sources: str = "",
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        error_message: str = "",
        error_stack: str = "",
        script_output: str = "",
        created_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
        num_objects: Optional[int] = None,
        size: Optional[int] = None,
        preview: Optional[list[dict]] = None,
        job_id: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> DatasetRecord:
        """Creates new dataset version."""

    @abstractmethod
    def remove_dataset(self, dataset: DatasetRecord) -> None:
        """Removes dataset."""

    @abstractmethod
    def update_dataset(self, dataset: DatasetRecord, **kwargs) -> DatasetRecord:
        """Updates dataset fields."""

    @abstractmethod
    def update_dataset_version(
        self, dataset: DatasetRecord, version: int, **kwargs
    ) -> DatasetVersion:
        """Updates dataset version fields."""

    @abstractmethod
    def remove_dataset_version(
        self, dataset: DatasetRecord, version: int
    ) -> DatasetRecord:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely.
        """

    @abstractmethod
    def list_datasets(self) -> Iterator[DatasetListRecord]:
        """Lists all datasets."""

    @abstractmethod
    def list_datasets_by_prefix(self, prefix: str) -> Iterator["DatasetListRecord"]:
        """Lists all datasets which names start with prefix."""

    @abstractmethod
    def get_dataset(self, name: str) -> DatasetRecord:
        """Gets a single dataset by name."""

    @abstractmethod
    def update_dataset_status(
        self,
        dataset: DatasetRecord,
        status: int,
        version: Optional[int] = None,
        error_message="",
        error_stack="",
        script_output="",
    ) -> DatasetRecord:
        """Updates dataset status and appropriate fields related to status."""

    #
    # Dataset dependencies
    #
    @abstractmethod
    def add_dataset_dependency(
        self,
        source_dataset_name: str,
        source_dataset_version: int,
        dataset_name: str,
        dataset_version: int,
    ) -> None:
        """Adds dataset dependency to dataset."""

    @abstractmethod
    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: int,
        new_source_dataset: Optional[DatasetRecord] = None,
        new_source_dataset_version: Optional[int] = None,
    ) -> None:
        """Updates dataset dependency source."""

    @abstractmethod
    def get_direct_dataset_dependencies(
        self, dataset: DatasetRecord, version: int
    ) -> list[Optional[DatasetDependency]]:
        """Gets direct dataset dependencies."""

    @abstractmethod
    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: Optional[int] = None
    ) -> None:
        """
        When we remove dataset, we need to clean up it's dependencies as well.
        """

    @abstractmethod
    def remove_dataset_dependants(
        self, dataset: DatasetRecord, version: Optional[int] = None
    ) -> None:
        """
        When we remove dataset, we need to clear its references in other dataset
        dependencies.
        """

    #
    # Jobs
    #

    def list_jobs_by_ids(self, ids: list[str], conn=None) -> Iterator["Job"]:
        raise NotImplementedError

    @abstractmethod
    def create_job(
        self,
        name: str,
        query: str,
        query_type: JobQueryType = JobQueryType.PYTHON,
        workers: int = 1,
        python_version: Optional[str] = None,
        params: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Creates a new job.
        Returns the job id.
        """

    @abstractmethod
    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set the status of the given job."""

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Returns the status of the given job."""

    @abstractmethod
    def set_job_and_dataset_status(
        self,
        job_id: str,
        job_status: JobStatus,
        dataset_status: DatasetStatus,
    ) -> None:
        """Set the status of the given job and dataset."""

    @abstractmethod
    def get_job_dataset_versions(self, job_id: str) -> list[tuple[str, int]]:
        """Returns dataset names and versions for the job."""
        raise NotImplementedError


class AbstractDBMetastore(AbstractMetastore):
    """
    Abstract Database Metastore class, to be implemented
    by any Database Adapters for a specific database system.
    This manages the storing, searching, and retrieval of indexed metadata,
    and has shared logic for all database systems currently in use.
    """

    DATASET_TABLE = "datasets"
    DATASET_VERSION_TABLE = "datasets_versions"
    DATASET_DEPENDENCY_TABLE = "datasets_dependencies"
    JOBS_TABLE = "jobs"

    db: "DatabaseEngine"

    def __init__(self, uri: Optional[StorageURI] = None):
        uri = uri or StorageURI("")
        super().__init__(uri)

    def close(self) -> None:
        """Closes any active database connections."""
        self.db.close()

    def cleanup_tables(self, temp_table_names: list[str]) -> None:
        """Cleanup temp tables."""

    @classmethod
    def _datasets_columns(cls) -> list["SchemaItem"]:
        """Datasets table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("labels", JSON, nullable=True),
            Column("status", Integer, nullable=False),
            Column("feature_schema", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True)),
            Column("finished_at", DateTime(timezone=True)),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("script_output", Text, nullable=False, default=""),
            Column("sources", Text, nullable=False, default=""),
            Column("query_script", Text, nullable=False, default=""),
            Column("schema", JSON, nullable=True),
        ]

    @cached_property
    def _dataset_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @cached_property
    def _dataset_list_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_columns()
            if c.name in self.dataset_list_class.__dataclass_fields__  # type: ignore [attr-defined]
        ]

    @classmethod
    def _datasets_versions_columns(cls) -> list["SchemaItem"]:
        """Datasets versions table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column(
                "dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("version", Integer, nullable=False),
            Column(
                "status",
                Integer,
                nullable=False,
            ),
            Column("feature_schema", JSON, nullable=True),
            Column("created_at", DateTime(timezone=True)),
            Column("finished_at", DateTime(timezone=True)),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("script_output", Text, nullable=False, default=""),
            Column("num_objects", BigInteger, nullable=True),
            Column("size", BigInteger, nullable=True),
            Column("preview", JSON, nullable=True),
            Column("sources", Text, nullable=False, default=""),
            Column("query_script", Text, nullable=False, default=""),
            Column("schema", JSON, nullable=True),
            Column("job_id", Text, nullable=True),
            UniqueConstraint("dataset_id", "version"),
        ]

    @cached_property
    def _dataset_version_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_versions_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @cached_property
    def _dataset_list_version_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._datasets_versions_columns()
            if c.name  # type: ignore [attr-defined]
            in self.dataset_list_version_class.__dataclass_fields__
        ]

    @classmethod
    def _datasets_dependencies_columns(cls) -> list["SchemaItem"]:
        """Datasets dependencies table columns."""
        return [
            Column("id", Integer, primary_key=True),
            # TODO remove when https://github.com/iterative/dvcx/issues/959 is done
            Column(
                "source_dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id"),
                nullable=False,
            ),
            Column(
                "source_dataset_version_id",
                Integer,
                ForeignKey(f"{cls.DATASET_VERSION_TABLE}.id"),
                nullable=True,
            ),
            # TODO remove when https://github.com/iterative/dvcx/issues/959 is done
            Column(
                "dataset_id",
                Integer,
                ForeignKey(f"{cls.DATASET_TABLE}.id"),
                nullable=True,
            ),
            Column(
                "dataset_version_id",
                Integer,
                ForeignKey(f"{cls.DATASET_VERSION_TABLE}.id"),
                nullable=True,
            ),
        ]

    #
    # Query Tables
    #
    @cached_property
    def _datasets(self) -> Table:
        return Table(self.DATASET_TABLE, self.db.metadata, *self._datasets_columns())

    @cached_property
    def _datasets_versions(self) -> Table:
        return Table(
            self.DATASET_VERSION_TABLE,
            self.db.metadata,
            *self._datasets_versions_columns(),
        )

    @cached_property
    def _datasets_dependencies(self) -> Table:
        return Table(
            self.DATASET_DEPENDENCY_TABLE,
            self.db.metadata,
            *self._datasets_dependencies_columns(),
        )

    #
    # Query Starters (These can be overridden by subclasses)
    #
    @abstractmethod
    def _datasets_insert(self) -> "Insert": ...

    def _datasets_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets.select()
        return select(*columns)

    def _datasets_update(self) -> "Update":
        return self._datasets.update()

    def _datasets_delete(self) -> "Delete":
        return self._datasets.delete()

    @abstractmethod
    def _datasets_versions_insert(self) -> "Insert": ...

    def _datasets_versions_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets_versions.select()
        return select(*columns)

    def _datasets_versions_update(self) -> "Update":
        return self._datasets_versions.update()

    def _datasets_versions_delete(self) -> "Delete":
        return self._datasets_versions.delete()

    @abstractmethod
    def _datasets_dependencies_insert(self) -> "Insert": ...

    def _datasets_dependencies_select(self, *columns) -> "Select":
        if not columns:
            return self._datasets_dependencies.select()
        return select(*columns)

    def _datasets_dependencies_update(self) -> "Update":
        return self._datasets_dependencies.update()

    def _datasets_dependencies_delete(self) -> "Delete":
        return self._datasets_dependencies.delete()

    #
    # Datasets
    #

    def create_dataset(
        self,
        name: str,
        status: int = DatasetStatus.CREATED,
        sources: Optional[list[str]] = None,
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
        **kwargs,  # TODO registered = True / False
    ) -> DatasetRecord:
        """Creates new dataset."""
        # TODO abstract this method and add registered = True based on kwargs
        query = self._datasets_insert().values(
            name=name,
            status=status,
            feature_schema=json.dumps(feature_schema or {}),
            created_at=datetime.now(timezone.utc),
            error_message="",
            error_stack="",
            script_output="",
            sources="\n".join(sources) if sources else "",
            query_script=query_script,
            schema=json.dumps(schema or {}),
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(index_elements=["name"])
        self.db.execute(query)

        return self.get_dataset(name)

    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: int,
        status: int,
        sources: str = "",
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        error_message: str = "",
        error_stack: str = "",
        script_output: str = "",
        created_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
        num_objects: Optional[int] = None,
        size: Optional[int] = None,
        preview: Optional[list[dict]] = None,
        job_id: Optional[str] = None,
        uuid: Optional[str] = None,
        conn=None,
    ) -> DatasetRecord:
        """Creates new dataset version."""
        if status in [DatasetStatus.COMPLETE, DatasetStatus.FAILED]:
            finished_at = finished_at or datetime.now(timezone.utc)
        else:
            finished_at = None

        query = self._datasets_versions_insert().values(
            dataset_id=dataset.id,
            uuid=uuid or str(uuid4()),
            version=version,
            status=status,
            feature_schema=json.dumps(feature_schema or {}),
            created_at=created_at or datetime.now(timezone.utc),
            finished_at=finished_at,
            error_message=error_message,
            error_stack=error_stack,
            script_output=script_output,
            sources=sources,
            query_script=query_script,
            schema=json.dumps(schema or {}),
            num_objects=num_objects,
            size=size,
            preview=json.dumps(preview or []),
            job_id=job_id or os.getenv("DATACHAIN_JOB_ID"),
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(
                index_elements=["dataset_id", "version"]
            )
        self.db.execute(query, conn=conn)

        return self.get_dataset(dataset.name, conn=conn)

    def remove_dataset(self, dataset: DatasetRecord) -> None:
        """Removes dataset."""
        d = self._datasets
        with self.db.transaction():
            self.remove_dataset_dependencies(dataset)
            self.remove_dataset_dependants(dataset)
            self.db.execute(self._datasets_delete().where(d.c.id == dataset.id))

    def update_dataset(
        self, dataset: DatasetRecord, conn=None, **kwargs
    ) -> DatasetRecord:
        """Updates dataset fields."""
        values = {}
        dataset_values = {}
        for field, value in kwargs.items():
            if field in self._dataset_fields[1:]:
                if field in ["labels", "schema"]:
                    values[field] = json.dumps(value) if value else None
                else:
                    values[field] = value
                if field == "schema":
                    dataset_values[field] = DatasetRecord.parse_schema(value)
                else:
                    dataset_values[field] = value

        if not values:
            # Nothing to update
            return dataset

        d = self._datasets
        self.db.execute(
            self._datasets_update().where(d.c.name == dataset.name).values(values),
            conn=conn,
        )  # type: ignore [attr-defined]

        result_ds = copy.deepcopy(dataset)
        result_ds.update(**dataset_values)
        return result_ds

    def update_dataset_version(
        self, dataset: DatasetRecord, version: int, conn=None, **kwargs
    ) -> DatasetVersion:
        """Updates dataset fields."""
        dataset_version = dataset.get_version(version)

        values = {}
        for field, value in kwargs.items():
            if field in self._dataset_version_fields[1:]:
                if field == "schema":
                    dataset_version.update(**{field: DatasetRecord.parse_schema(value)})
                    values[field] = json.dumps(value) if value else None
                elif field == "feature_schema":
                    values[field] = json.dumps(value) if value else None
                elif field == "preview" and isinstance(value, list):
                    values[field] = json.dumps(value, cls=JSONSerialize)
                else:
                    values[field] = value
                    dataset_version.update(**{field: value})

        if not values:
            # Nothing to update
            return dataset_version

        dv = self._datasets_versions
        self.db.execute(
            self._datasets_versions_update()
            .where(dv.c.id == dataset_version.id)
            .values(values),
            conn=conn,
        )  # type: ignore [attr-defined]

        return dataset_version

    def _parse_dataset(self, rows) -> Optional[DatasetRecord]:
        versions = [self.dataset_class.parse(*r) for r in rows]
        if not versions:
            return None
        return reduce(lambda ds, version: ds.merge_versions(version), versions)

    def _parse_list_dataset(self, rows) -> Optional[DatasetListRecord]:
        versions = [self.dataset_list_class.parse(*r) for r in rows]
        if not versions:
            return None
        return reduce(lambda ds, version: ds.merge_versions(version), versions)

    def _parse_dataset_list(self, rows) -> Iterator["DatasetListRecord"]:
        # grouping rows by dataset id
        for _, g in groupby(rows, lambda r: r[0]):
            dataset = self._parse_list_dataset(list(g))
            if dataset:
                yield dataset

    def _get_dataset_query(
        self,
        dataset_fields: list[str],
        dataset_version_fields: list[str],
        isouter: bool = True,
    ):
        if not (
            self.db.has_table(self._datasets.name)
            and self.db.has_table(self._datasets_versions.name)
        ):
            raise TableMissingError

        d = self._datasets
        dv = self._datasets_versions

        query = self._datasets_select(
            *(getattr(d.c, f) for f in dataset_fields),
            *(getattr(dv.c, f) for f in dataset_version_fields),
        )
        j = d.join(dv, d.c.id == dv.c.dataset_id, isouter=isouter)
        return query.select_from(j)

    def _base_dataset_query(self):
        return self._get_dataset_query(
            self._dataset_fields, self._dataset_version_fields
        )

    def _base_list_datasets_query(self):
        return self._get_dataset_query(
            self._dataset_list_fields, self._dataset_list_version_fields, isouter=False
        )

    def list_datasets(self) -> Iterator["DatasetListRecord"]:
        """Lists all datasets."""
        query = self._base_list_datasets_query().order_by(
            self._datasets.c.name, self._datasets_versions.c.version
        )
        yield from self._parse_dataset_list(self.db.execute(query))

    def list_datasets_by_prefix(
        self, prefix: str, conn=None
    ) -> Iterator["DatasetListRecord"]:
        query = self._base_list_datasets_query()
        query = query.where(self._datasets.c.name.startswith(prefix))
        yield from self._parse_dataset_list(self.db.execute(query))

    def get_dataset(self, name: str, conn=None) -> DatasetRecord:
        """Gets a single dataset by name"""
        d = self._datasets
        query = self._base_dataset_query()
        query = query.where(d.c.name == name)  # type: ignore [attr-defined]
        ds = self._parse_dataset(self.db.execute(query, conn=conn))
        if not ds:
            raise DatasetNotFoundError(f"Dataset {name} not found.")
        return ds

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: int
    ) -> DatasetRecord:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely
        """
        if not dataset.has_version(version):
            raise DatasetNotFoundError(
                f"Dataset {dataset.name} version {version} not found."
            )

        self.remove_dataset_dependencies(dataset, version)
        self.remove_dataset_dependants(dataset, version)

        d = self._datasets
        dv = self._datasets_versions
        self.db.execute(
            self._datasets_versions_delete().where(
                (dv.c.dataset_id == dataset.id) & (dv.c.version == version)
            )
        )

        if dataset.versions and len(dataset.versions) == 1:
            # had only one version, fully deleting dataset
            self.db.execute(self._datasets_delete().where(d.c.id == dataset.id))

        dataset.remove_version(version)
        return dataset

    def update_dataset_status(
        self,
        dataset: DatasetRecord,
        status: int,
        version: Optional[int] = None,
        error_message="",
        error_stack="",
        script_output="",
        conn=None,
    ) -> DatasetRecord:
        """
        Updates dataset status and appropriate fields related to status
        It also updates version if specified.
        """
        update_data: dict[str, Any] = {"status": status}
        if status in [DatasetStatus.COMPLETE, DatasetStatus.FAILED]:
            # if in final state, updating finished_at datetime
            update_data["finished_at"] = datetime.now(timezone.utc)
            if script_output:
                update_data["script_output"] = script_output

        if status == DatasetStatus.FAILED:
            update_data["error_message"] = error_message
            update_data["error_stack"] = error_stack

        self.update_dataset(dataset, conn=conn, **update_data)

        if version:
            self.update_dataset_version(dataset, version, conn=conn, **update_data)

        return dataset

    #
    # Dataset dependencies
    #
    def add_dataset_dependency(
        self,
        source_dataset_name: str,
        source_dataset_version: int,
        dataset_name: str,
        dataset_version: int,
    ) -> None:
        """Adds dataset dependency to dataset."""
        source_dataset = self.get_dataset(source_dataset_name)
        dataset = self.get_dataset(dataset_name)

        self.db.execute(
            self._datasets_dependencies_insert().values(
                source_dataset_id=source_dataset.id,
                source_dataset_version_id=(
                    source_dataset.get_version(source_dataset_version).id
                ),
                dataset_id=dataset.id,
                dataset_version_id=dataset.get_version(dataset_version).id,
            )
        )

    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: int,
        new_source_dataset: Optional[DatasetRecord] = None,
        new_source_dataset_version: Optional[int] = None,
    ) -> None:
        dd = self._datasets_dependencies

        if not new_source_dataset:
            new_source_dataset = source_dataset

        q = self._datasets_dependencies_update().where(
            dd.c.source_dataset_id == source_dataset.id
        )
        q = q.where(
            dd.c.source_dataset_version_id
            == source_dataset.get_version(source_dataset_version).id
        )

        data = {"source_dataset_id": new_source_dataset.id}
        if new_source_dataset_version:
            data["source_dataset_version_id"] = new_source_dataset.get_version(
                new_source_dataset_version
            ).id

        q = q.values(**data)
        self.db.execute(q)

    @abstractmethod
    def _dataset_dependencies_select_columns(self) -> list["SchemaItem"]:
        """
        Returns a list of columns to select in a query for fetching dataset dependencies
        """

    def get_direct_dataset_dependencies(
        self, dataset: DatasetRecord, version: int
    ) -> list[Optional[DatasetDependency]]:
        d = self._datasets
        dd = self._datasets_dependencies
        dv = self._datasets_versions

        dataset_version = dataset.get_version(version)

        select_cols = self._dataset_dependencies_select_columns()

        query = (
            self._datasets_dependencies_select(*select_cols)
            .select_from(
                dd.join(d, dd.c.dataset_id == d.c.id, isouter=True).join(
                    dv, dd.c.dataset_version_id == dv.c.id, isouter=True
                )
            )
            .where(
                (dd.c.source_dataset_id == dataset.id)
                & (dd.c.source_dataset_version_id == dataset_version.id)
            )
        )
        if version:
            dataset_version = dataset.get_version(version)
            query = query.where(dd.c.source_dataset_version_id == dataset_version.id)

        return [self.dependency_class.parse(*r) for r in self.db.execute(query)]

    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: Optional[int] = None
    ) -> None:
        """
        When we remove dataset, we need to clean up it's dependencies as well
        """
        dd = self._datasets_dependencies

        q = self._datasets_dependencies_delete().where(
            dd.c.source_dataset_id == dataset.id
        )

        if version:
            q = q.where(
                dd.c.source_dataset_version_id == dataset.get_version(version).id
            )

        self.db.execute(q)

    def remove_dataset_dependants(
        self, dataset: DatasetRecord, version: Optional[int] = None
    ) -> None:
        """
        When we remove dataset, we need to clear its references in other dataset
        dependencies
        """
        dd = self._datasets_dependencies

        q = self._datasets_dependencies_update().where(dd.c.dataset_id == dataset.id)
        if version:
            q = q.where(dd.c.dataset_version_id == dataset.get_version(version).id)

        q = q.values(dataset_id=None, dataset_version_id=None)

        self.db.execute(q)

    #
    # Jobs
    #

    @staticmethod
    def _jobs_columns() -> "list[SchemaItem]":
        return [
            Column(
                "id",
                Text,
                default=uuid4,
                primary_key=True,
                nullable=False,
            ),
            Column("name", Text, nullable=False, default=""),
            Column("status", Integer, nullable=False, default=JobStatus.CREATED),
            # When this Job was created
            Column("created_at", DateTime(timezone=True), nullable=False),
            # When this Job finished (or failed)
            Column("finished_at", DateTime(timezone=True), nullable=True),
            # This is the workers value from query settings, and determines both
            # the default and maximum number of workers for distributed UDFs.
            Column("query", Text, nullable=False, default=""),
            Column(
                "query_type",
                Integer,
                nullable=False,
                default=JobQueryType.PYTHON,
            ),
            Column("workers", Integer, nullable=False, default=1),
            Column("python_version", Text, nullable=True),
            Column("error_message", Text, nullable=False, default=""),
            Column("error_stack", Text, nullable=False, default=""),
            Column("params", JSON, nullable=False),
            Column("metrics", JSON, nullable=False),
        ]

    @cached_property
    def _job_fields(self) -> list[str]:
        return [c.name for c in self._jobs_columns() if c.name]  # type: ignore[attr-defined]

    @cached_property
    def _jobs(self) -> "Table":
        return Table(self.JOBS_TABLE, self.db.metadata, *self._jobs_columns())

    @abstractmethod
    def _jobs_insert(self) -> "Insert": ...

    def _jobs_select(self, *columns) -> "Select":
        if not columns:
            return self._jobs.select()
        return select(*columns)

    def _jobs_update(self, *where) -> "Update":
        if not where:
            return self._jobs.update()
        return self._jobs.update().where(*where)

    def _parse_job(self, rows) -> Job:
        return self.job_class.parse(*rows)

    def _parse_jobs(self, rows) -> Iterator["Job"]:
        for _, g in groupby(rows, lambda r: r[0]):
            yield self._parse_job(*list(g))

    def _jobs_query(self):
        return self._jobs_select(*[getattr(self._jobs.c, f) for f in self._job_fields])

    def list_jobs_by_ids(self, ids: list[str], conn=None) -> Iterator["Job"]:
        """List jobs by ids."""
        query = self._jobs_query().where(self._jobs.c.id.in_(ids))
        yield from self._parse_jobs(self.db.execute(query, conn=conn))

    def create_job(
        self,
        name: str,
        query: str,
        query_type: JobQueryType = JobQueryType.PYTHON,
        workers: int = 1,
        python_version: Optional[str] = None,
        params: Optional[dict[str, str]] = None,
        conn: Optional[Any] = None,
    ) -> str:
        """
        Creates a new job.
        Returns the job id.
        """
        job_id = str(uuid4())
        self.db.execute(
            self._jobs_insert().values(
                id=job_id,
                name=name,
                status=JobStatus.CREATED,
                created_at=datetime.now(timezone.utc),
                query=query,
                query_type=query_type.value,
                workers=workers,
                python_version=python_version,
                error_message="",
                error_stack="",
                params=json.dumps(params or {}),
                metrics=json.dumps({}),
            ),
            conn=conn,
        )
        return job_id

    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        conn: Optional[Any] = None,
    ) -> None:
        """Set the status of the given job."""
        values: dict = {"status": status.value}
        if status.value in JobStatus.finished():
            values["finished_at"] = datetime.now(timezone.utc)
        if error_message:
            values["error_message"] = error_message
        if error_stack:
            values["error_stack"] = error_stack
        if metrics:
            values["metrics"] = json.dumps(metrics)
        self.db.execute(
            self._jobs_update(self._jobs.c.id == job_id).values(**values),
            conn=conn,
        )

    def get_job_status(
        self,
        job_id: str,
        conn: Optional[Any] = None,
    ) -> Optional[JobStatus]:
        """Returns the status of the given job."""
        results = list(
            self.db.execute(
                self._jobs_select(self._jobs.c.status).where(self._jobs.c.id == job_id),
                conn=conn,
            ),
        )
        if not results:
            return None
        return results[0][0]

    def set_job_and_dataset_status(
        self,
        job_id: str,
        job_status: JobStatus,
        dataset_status: DatasetStatus,
    ) -> None:
        """Set the status of the given job and dataset."""
        with self.db.transaction() as conn:
            self.set_job_status(job_id, status=job_status, conn=conn)
            dv = self._datasets_versions
            query = (
                self._datasets_versions_update()
                .where(
                    (dv.c.job_id == job_id) & (dv.c.status != DatasetStatus.COMPLETE)
                )
                .values(status=dataset_status)
            )
            self.db.execute(query, conn=conn)  # type: ignore[attr-defined]

    def get_job_dataset_versions(self, job_id: str) -> list[tuple[str, int]]:
        """Returns dataset names and versions for the job."""
        dv = self._datasets_versions
        ds = self._datasets

        join_condition = dv.c.dataset_id == ds.c.id

        query = (
            self._datasets_versions_select(ds.c.name, dv.c.version)
            .select_from(dv.join(ds, join_condition))
            .where(dv.c.job_id == job_id)
        )

        return list(self.db.execute(query))
