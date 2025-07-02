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
    DatasetVersionNotFoundError,
    NamespaceNotFoundError,
    ProjectNotFoundError,
    TableMissingError,
)
from datachain.job import Job
from datachain.namespace import Namespace
from datachain.project import Project
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
    namespace_class: type[Namespace] = Namespace
    project_class: type[Project] = Project
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
    # Namespaces
    #

    @property
    @abstractmethod
    def default_namespace_name(self):
        """Gets default namespace name"""

    @property
    def system_namespace_name(self):
        return Namespace.system()

    @abstractmethod
    def create_namespace(
        self,
        name: str,
        description: Optional[str] = None,
        uuid: Optional[str] = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Namespace:
        """Creates new namespace"""

    @abstractmethod
    def get_namespace(self, name: str, conn=None) -> Namespace:
        """Gets a single namespace by name"""

    @abstractmethod
    def list_namespaces(self, conn=None) -> list[Namespace]:
        """Gets a list of all namespaces"""

    @property
    @abstractmethod
    def is_studio(self) -> bool:
        """Returns True if this code is ran in Studio"""

    def is_local_dataset(self, dataset_namespace: str) -> bool:
        """
        Returns True if this is local dataset i.e. not pulled from Studio but
        created locally. This is False if we ran code in CLI mode but using dataset
        names that are present in Studio.
        """
        return self.is_studio or dataset_namespace == Namespace.default()

    @property
    def namespace_allowed_to_create(self):
        return self.is_studio

    #
    # Projects
    #

    @property
    @abstractmethod
    def default_project_name(self):
        """Gets default project name"""

    @property
    def listing_project_name(self):
        return Project.listing()

    @cached_property
    def default_project(self) -> Project:
        return self.get_project(
            self.default_project_name, self.default_namespace_name, create=True
        )

    @cached_property
    def listing_project(self) -> Project:
        return self.get_project(self.listing_project_name, self.system_namespace_name)

    @abstractmethod
    def create_project(
        self,
        namespace_name: str,
        name: str,
        description: Optional[str] = None,
        uuid: Optional[str] = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Project:
        """Creates new project in specific namespace"""

    @abstractmethod
    def get_project(
        self, name: str, namespace_name: str, create: bool = False, conn=None
    ) -> Project:
        """
        Gets a single project inside some namespace by name.
        It also creates project if not found and create flag is set to True.
        """

    @abstractmethod
    def list_projects(self, namespace_id: Optional[int], conn=None) -> list[Project]:
        """Gets list of projects in some namespace or in general (in all namespaces)"""

    @property
    def project_allowed_to_create(self):
        return self.is_studio

    #
    # Datasets
    #
    @abstractmethod
    def create_dataset(
        self,
        name: str,
        project_id: Optional[int] = None,
        status: int = DatasetStatus.CREATED,
        sources: Optional[list[str]] = None,
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
        description: Optional[str] = None,
        attrs: Optional[list[str]] = None,
    ) -> DatasetRecord:
        """Creates new dataset."""

    @abstractmethod
    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: str,
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
        self, dataset: DatasetRecord, version: str, **kwargs
    ) -> DatasetVersion:
        """Updates dataset version fields."""

    @abstractmethod
    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str
    ) -> DatasetRecord:
        """
        Deletes one single dataset version.
        If it was last version, it removes dataset completely.
        """

    @abstractmethod
    def list_datasets(
        self, project_id: Optional[int] = None
    ) -> Iterator[DatasetListRecord]:
        """Lists all datasets in some project or in all projects."""

    @abstractmethod
    def list_datasets_by_prefix(
        self, prefix: str, project_id: Optional[int] = None
    ) -> Iterator["DatasetListRecord"]:
        """
        Lists all datasets which names start with prefix in some project or in all
        projects.
        """

    @abstractmethod
    def get_dataset(self, name: str, project_id: Optional[int] = None) -> DatasetRecord:
        """Gets a single dataset by name."""

    @abstractmethod
    def update_dataset_status(
        self,
        dataset: DatasetRecord,
        status: int,
        version: Optional[str] = None,
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
        source_dataset: "DatasetRecord",
        source_dataset_version: str,
        dep_dataset: "DatasetRecord",
        dep_dataset_version: str,
    ) -> None:
        """Adds dataset dependency to dataset."""

    @abstractmethod
    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: str,
        new_source_dataset: Optional[DatasetRecord] = None,
        new_source_dataset_version: Optional[str] = None,
    ) -> None:
        """Updates dataset dependency source."""

    @abstractmethod
    def get_direct_dataset_dependencies(
        self, dataset: DatasetRecord, version: str
    ) -> list[Optional[DatasetDependency]]:
        """Gets direct dataset dependencies."""

    @abstractmethod
    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: Optional[str] = None
    ) -> None:
        """
        When we remove dataset, we need to clean up it's dependencies as well.
        """

    @abstractmethod
    def remove_dataset_dependants(
        self, dataset: DatasetRecord, version: Optional[str] = None
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
        status: JobStatus = JobStatus.CREATED,
        workers: int = 1,
        python_version: Optional[str] = None,
        params: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Creates a new job.
        Returns the job id.
        """

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Returns the job with the given ID."""

    @abstractmethod
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        finished_at: Optional[datetime] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Optional["Job"]:
        """Updates job fields."""

    @abstractmethod
    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
    ) -> None:
        """Set the status of the given job."""

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Returns the status of the given job."""


class AbstractDBMetastore(AbstractMetastore):
    """
    Abstract Database Metastore class, to be implemented
    by any Database Adapters for a specific database system.
    This manages the storing, searching, and retrieval of indexed metadata,
    and has shared logic for all database systems currently in use.
    """

    NAMESPACE_TABLE = "namespaces"
    PROJECT_TABLE = "projects"
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
    def _namespaces_columns(cls) -> list["SchemaItem"]:
        """Namespace table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("created_at", DateTime(timezone=True)),
        ]

    @cached_property
    def _namespaces_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._namespaces_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @classmethod
    def _projects_columns(cls) -> list["SchemaItem"]:
        """Project table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column("uuid", Text, nullable=False, default=uuid4()),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("created_at", DateTime(timezone=True)),
            Column(
                "namespace_id",
                Integer,
                ForeignKey(f"{cls.NAMESPACE_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            UniqueConstraint("namespace_id", "name"),
        ]

    @cached_property
    def _projects_fields(self) -> list[str]:
        return [
            c.name  # type: ignore [attr-defined]
            for c in self._projects_columns()
            if c.name  # type: ignore [attr-defined]
        ]

    @classmethod
    def _datasets_columns(cls) -> list["SchemaItem"]:
        """Datasets table columns."""
        return [
            Column("id", Integer, primary_key=True),
            Column(
                "project_id",
                Integer,
                ForeignKey(f"{cls.PROJECT_TABLE}.id", ondelete="CASCADE"),
                nullable=False,
            ),
            Column("name", Text, nullable=False),
            Column("description", Text),
            Column("attrs", JSON, nullable=True),
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
            Column("version", Text, nullable=False, default="1.0.0"),
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
    def _namespaces(self) -> Table:
        return Table(
            self.NAMESPACE_TABLE, self.db.metadata, *self._namespaces_columns()
        )

    @cached_property
    def _projects(self) -> Table:
        return Table(self.PROJECT_TABLE, self.db.metadata, *self._projects_columns())

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
    def _namespaces_insert(self) -> "Insert": ...

    def _namespaces_select(self, *columns) -> "Select":
        if not columns:
            return self._namespaces.select()
        return select(*columns)

    def _namespaces_update(self) -> "Update":
        return self._namespaces.update()

    def _namespaces_delete(self) -> "Delete":
        return self._namespaces.delete()

    @abstractmethod
    def _projects_insert(self) -> "Insert": ...

    def _projects_select(self, *columns) -> "Select":
        if not columns:
            return self._projects.select()
        return select(*columns)

    def _projects_update(self) -> "Update":
        return self._projects.update()

    def _projects_delete(self) -> "Delete":
        return self._projects.delete()

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
    # Namespaces
    #

    def create_namespace(
        self,
        name: str,
        description: Optional[str] = None,
        uuid: Optional[str] = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Namespace:
        if validate:
            Namespace.validate_name(name)
        query = self._namespaces_insert().values(
            name=name,
            uuid=uuid or str(uuid4()),
            created_at=datetime.now(timezone.utc),
            description=description,
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(index_elements=["name"])
        self.db.execute(query)

        return self.get_namespace(name)

    def get_namespace(self, name: str, conn=None) -> Namespace:
        """Gets a single namespace by name"""
        n = self._namespaces

        query = self._namespaces_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
        ).where(n.c.name == name)
        rows = list(self.db.execute(query, conn=conn))
        if not rows:
            raise NamespaceNotFoundError(f"Namespace {name} not found.")
        return self.namespace_class.parse(*rows[0])

    def list_namespaces(self, conn=None) -> list[Namespace]:
        """Gets a list of all namespaces"""
        n = self._namespaces

        query = self._namespaces_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
        )
        rows = list(self.db.execute(query, conn=conn))

        return [self.namespace_class.parse(*r) for r in rows]

    #
    # Projects
    #

    def create_project(
        self,
        namespace_name: str,
        name: str,
        description: Optional[str] = None,
        uuid: Optional[str] = None,
        ignore_if_exists: bool = True,
        validate: bool = True,
        **kwargs,
    ) -> Project:
        if validate:
            Project.validate_name(name)
        try:
            namespace = self.get_namespace(namespace_name)
        except NamespaceNotFoundError:
            namespace = self.create_namespace(namespace_name, validate=validate)

        query = self._projects_insert().values(
            namespace_id=namespace.id,
            uuid=uuid or str(uuid4()),
            name=name,
            created_at=datetime.now(timezone.utc),
            description=description,
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(
                index_elements=["namespace_id", "name"]
            )
        self.db.execute(query)

        return self.get_project(name, namespace.name)

    def _is_listing_project(self, project_name: str, namespace_name: str) -> bool:
        return (
            project_name == self.listing_project_name
            and namespace_name == self.system_namespace_name
        )

    def _is_default_project(self, project_name: str, namespace_name: str) -> bool:
        return (
            project_name == self.default_project_name
            and namespace_name == self.default_namespace_name
        )

    def get_project(
        self, name: str, namespace_name: str, create: bool = False, conn=None
    ) -> Project:
        """Gets a single project inside some namespace by name"""
        n = self._namespaces
        p = self._projects
        validate = True

        if self._is_listing_project(name, namespace_name) or self._is_default_project(
            name, namespace_name
        ):
            # we are always creating default and listing projects if they don't exist
            create = True
            validate = False

        query = self._projects_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
            *(getattr(p.c, f) for f in self._projects_fields),
        )
        query = query.select_from(n.join(p, n.c.id == p.c.namespace_id)).where(
            p.c.name == name, n.c.name == namespace_name
        )

        rows = list(self.db.execute(query, conn=conn))
        if not rows:
            if create:
                return self.create_project(namespace_name, name, validate=validate)
            raise ProjectNotFoundError(
                f"Project {name} in namespace {namespace_name} not found."
            )
        return self.project_class.parse(*rows[0])

    def list_projects(self, namespace_id: Optional[int], conn=None) -> list[Project]:
        """
        Gets a list of projects inside some namespace, or in all namespaces
        """
        n = self._namespaces
        p = self._projects

        query = self._projects_select(
            *(getattr(n.c, f) for f in self._namespaces_fields),
            *(getattr(p.c, f) for f in self._projects_fields),
        )
        query = query.select_from(n.join(p, n.c.id == p.c.namespace_id))

        if namespace_id:
            query = query.where(n.c.id == namespace_id)

        rows = list(self.db.execute(query, conn=conn))

        return [self.project_class.parse(*r) for r in rows]

    #
    # Datasets
    #

    def create_dataset(
        self,
        name: str,
        project_id: Optional[int] = None,
        status: int = DatasetStatus.CREATED,
        sources: Optional[list[str]] = None,
        feature_schema: Optional[dict] = None,
        query_script: str = "",
        schema: Optional[dict[str, Any]] = None,
        ignore_if_exists: bool = False,
        description: Optional[str] = None,
        attrs: Optional[list[str]] = None,
        **kwargs,  # TODO registered = True / False
    ) -> DatasetRecord:
        """Creates new dataset."""
        project_id = project_id or self.default_project.id

        query = self._datasets_insert().values(
            name=name,
            project_id=project_id,
            status=status,
            feature_schema=json.dumps(feature_schema or {}),
            created_at=datetime.now(timezone.utc),
            error_message="",
            error_stack="",
            script_output="",
            sources="\n".join(sources) if sources else "",
            query_script=query_script,
            schema=json.dumps(schema or {}),
            description=description,
            attrs=json.dumps(attrs or []),
        )
        if ignore_if_exists and hasattr(query, "on_conflict_do_nothing"):
            # SQLite and PostgreSQL both support 'on_conflict_do_nothing',
            # but generic SQL does not
            query = query.on_conflict_do_nothing(index_elements=["project_id", "name"])
        self.db.execute(query)

        return self.get_dataset(name, project_id)

    def create_dataset_version(  # noqa: PLR0913
        self,
        dataset: DatasetRecord,
        version: str,
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

        return self.get_dataset(dataset.name, dataset.project.id, conn=conn)

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
        values: dict[str, Any] = {}
        dataset_values: dict[str, Any] = {}
        for field, value in kwargs.items():
            if field in ("id", "created_at") or field not in self._dataset_fields:
                continue  # these fields are read-only or not applicable

            if value is None and field in ("name", "status", "sources", "query_script"):
                raise ValueError(f"Field {field} cannot be None")
            if field == "name" and not value:
                raise ValueError("name cannot be empty")

            if field == "attrs":
                if value is None:
                    values[field] = None
                else:
                    values[field] = json.dumps(value)
                dataset_values[field] = value
            elif field == "schema":
                if value is None:
                    values[field] = None
                    dataset_values[field] = None
                else:
                    values[field] = json.dumps(value)
                    dataset_values[field] = DatasetRecord.parse_schema(value)
            else:
                values[field] = value
                dataset_values[field] = value

        if not values:
            return dataset  # nothing to update

        d = self._datasets
        self.db.execute(
            self._datasets_update().where(d.c.name == dataset.name).values(values),
            conn=conn,
        )  # type: ignore [attr-defined]

        result_ds = copy.deepcopy(dataset)
        result_ds.update(**dataset_values)
        return result_ds

    def update_dataset_version(
        self, dataset: DatasetRecord, version: str, conn=None, **kwargs
    ) -> DatasetVersion:
        """Updates dataset fields."""
        values: dict[str, Any] = {}
        version_values: dict[str, Any] = {}
        for field, value in kwargs.items():
            if (
                field in ("id", "created_at")
                or field not in self._dataset_version_fields
            ):
                continue  # these fields are read-only or not applicable

            if value is None and field in (
                "status",
                "sources",
                "query_script",
                "error_message",
                "error_stack",
                "script_output",
                "uuid",
            ):
                raise ValueError(f"Field {field} cannot be None")

            if field == "schema":
                values[field] = json.dumps(value) if value else None
                version_values[field] = (
                    DatasetRecord.parse_schema(value) if value else None
                )
            elif field == "feature_schema":
                if value is None:
                    values[field] = None
                else:
                    values[field] = json.dumps(value)
                version_values[field] = value
            elif field == "preview":
                if value is None:
                    values[field] = None
                elif not isinstance(value, list):
                    raise ValueError(
                        f"Field '{field}' must be a list, got {type(value).__name__}"
                    )
                else:
                    values[field] = json.dumps(value, cls=JSONSerialize)
                version_values["_preview_data"] = value
            else:
                values[field] = value
                version_values[field] = value

        if not values:
            return dataset.get_version(version)

        dv = self._datasets_versions
        self.db.execute(
            self._datasets_versions_update()
            .where(dv.c.dataset_id == dataset.id, dv.c.version == version)
            .values(values),
            conn=conn,
        )  # type: ignore [attr-defined]

        for v in dataset.versions:
            if v.version == version:
                v.update(**version_values)
                return v

        raise DatasetVersionNotFoundError(
            f"Dataset {dataset.name} does not have version {version}"
        )

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
        for _, g in groupby(rows, lambda r: r[11]):
            dataset = self._parse_list_dataset(list(g))
            if dataset:
                yield dataset

    def _get_dataset_query(
        self,
        namespace_fields: list[str],
        project_fields: list[str],
        dataset_fields: list[str],
        dataset_version_fields: list[str],
        isouter: bool = True,
    ) -> "Select":
        if not (
            self.db.has_table(self._datasets.name)
            and self.db.has_table(self._datasets_versions.name)
        ):
            raise TableMissingError

        n = self._namespaces
        p = self._projects
        d = self._datasets
        dv = self._datasets_versions

        query = self._datasets_select(
            *(getattr(n.c, f) for f in namespace_fields),
            *(getattr(p.c, f) for f in project_fields),
            *(getattr(d.c, f) for f in dataset_fields),
            *(getattr(dv.c, f) for f in dataset_version_fields),
        )
        j = (
            n.join(p, n.c.id == p.c.namespace_id)
            .join(d, p.c.id == d.c.project_id)
            .join(dv, d.c.id == dv.c.dataset_id, isouter=isouter)
        )
        return query.select_from(j)

    def _base_dataset_query(self) -> "Select":
        return self._get_dataset_query(
            self._namespaces_fields,
            self._projects_fields,
            self._dataset_fields,
            self._dataset_version_fields,
        )

    def _base_list_datasets_query(self) -> "Select":
        return self._get_dataset_query(
            self._namespaces_fields,
            self._projects_fields,
            self._dataset_list_fields,
            self._dataset_list_version_fields,
            isouter=False,
        )

    def list_datasets(
        self, project_id: Optional[int] = None
    ) -> Iterator["DatasetListRecord"]:
        """Lists all datasets."""
        d = self._datasets
        query = self._base_list_datasets_query().order_by(
            self._datasets.c.name, self._datasets_versions.c.version
        )
        if project_id:
            query = query.where(d.c.project_id == project_id)
        yield from self._parse_dataset_list(self.db.execute(query))

    def list_datasets_by_prefix(
        self, prefix: str, project_id: Optional[int] = None, conn=None
    ) -> Iterator["DatasetListRecord"]:
        d = self._datasets
        query = self._base_list_datasets_query()
        if project_id:
            query = query.where(d.c.project_id == project_id)
        query = query.where(self._datasets.c.name.startswith(prefix))
        yield from self._parse_dataset_list(self.db.execute(query))

    def get_dataset(
        self,
        name: str,  # normal, not full dataset name
        project_id: Optional[int] = None,
        conn=None,
    ) -> DatasetRecord:
        """
        Gets a single dataset in project by dataset name.
        """
        project_id = project_id or self.default_project.id
        d = self._datasets
        query = self._base_dataset_query()
        query = query.where(d.c.name == name, d.c.project_id == project_id)  # type: ignore [attr-defined]
        ds = self._parse_dataset(self.db.execute(query, conn=conn))
        if not ds:
            raise DatasetNotFoundError(
                f"Dataset {name} not found in project {project_id}"
            )
        return ds

    def remove_dataset_version(
        self, dataset: DatasetRecord, version: str
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
        version: Optional[str] = None,
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

        dataset = self.update_dataset(dataset, conn=conn, **update_data)

        if version:
            self.update_dataset_version(dataset, version, conn=conn, **update_data)

        return dataset

    #
    # Dataset dependencies
    #
    def add_dataset_dependency(
        self,
        source_dataset: "DatasetRecord",
        source_dataset_version: str,
        dep_dataset: "DatasetRecord",
        dep_dataset_version: str,
    ) -> None:
        """Adds dataset dependency to dataset."""
        self.db.execute(
            self._datasets_dependencies_insert().values(
                source_dataset_id=source_dataset.id,
                source_dataset_version_id=(
                    source_dataset.get_version(source_dataset_version).id
                ),
                dataset_id=dep_dataset.id,
                dataset_version_id=dep_dataset.get_version(dep_dataset_version).id,
            )
        )

    def update_dataset_dependency_source(
        self,
        source_dataset: DatasetRecord,
        source_dataset_version: str,
        new_source_dataset: Optional[DatasetRecord] = None,
        new_source_dataset_version: Optional[str] = None,
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
        self, dataset: DatasetRecord, version: str
    ) -> list[Optional[DatasetDependency]]:
        n = self._namespaces
        p = self._projects
        d = self._datasets
        dd = self._datasets_dependencies
        dv = self._datasets_versions

        dataset_version = dataset.get_version(version)

        select_cols = self._dataset_dependencies_select_columns()

        query = (
            self._datasets_dependencies_select(*select_cols)
            .select_from(
                dd.join(d, dd.c.dataset_id == d.c.id, isouter=True)
                .join(dv, dd.c.dataset_version_id == dv.c.id, isouter=True)
                .join(p, d.c.project_id == p.c.id, isouter=True)
                .join(n, p.c.namespace_id == n.c.id, isouter=True)
            )
            .where(
                (dd.c.source_dataset_id == dataset.id)
                & (dd.c.source_dataset_version_id == dataset_version.id)
            )
        )

        return [self.dependency_class.parse(*r) for r in self.db.execute(query)]

    def remove_dataset_dependencies(
        self, dataset: DatasetRecord, version: Optional[str] = None
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
        self, dataset: DatasetRecord, version: Optional[str] = None
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
        status: JobStatus = JobStatus.CREATED,
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
                status=status,
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

    def get_job(self, job_id: str, conn=None) -> Optional[Job]:
        """Returns the job with the given ID."""
        query = self._jobs_select(self._jobs).where(self._jobs.c.id == job_id)
        results = list(self.db.execute(query, conn=conn))
        if not results:
            return None
        return self._parse_job(results[0])

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        finished_at: Optional[datetime] = None,
        metrics: Optional[dict[str, Any]] = None,
        conn: Optional[Any] = None,
    ) -> Optional["Job"]:
        """Updates job fields."""
        values: dict = {}
        if status is not None:
            values["status"] = status
        if error_message is not None:
            values["error_message"] = error_message
        if error_stack is not None:
            values["error_stack"] = error_stack
        if finished_at is not None:
            values["finished_at"] = finished_at
        if metrics:
            values["metrics"] = json.dumps(metrics)

        if values:
            j = self._jobs
            self.db.execute(
                self._jobs_update().where(j.c.id == job_id).values(**values),
                conn=conn,
            )  # type: ignore [attr-defined]

        return self.get_job(job_id, conn=conn)

    def set_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        error_stack: Optional[str] = None,
        conn: Optional[Any] = None,
    ) -> None:
        """Set the status of the given job."""
        values: dict = {"status": status}
        if status in JobStatus.finished():
            values["finished_at"] = datetime.now(timezone.utc)
        if error_message:
            values["error_message"] = error_message
        if error_stack:
            values["error_stack"] = error_stack
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
