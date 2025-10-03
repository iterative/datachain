import atexit
import gc
import logging
import os
import re
import sys
import traceback
from typing import TYPE_CHECKING, Callable, ClassVar, Optional
from uuid import uuid4

from datachain.catalog import get_catalog
from datachain.data_storage import JobQueryType, JobStatus
from datachain.error import JobNotFoundError, TableMissingError

if TYPE_CHECKING:
    from datachain.catalog import Catalog
    from datachain.dataset import DatasetRecord
    from datachain.job import Job

logger = logging.getLogger("datachain")


class Session:
    """
    Session is a context that keeps track of temporary DataChain datasets for a proper
    cleanup. By default, a global session is created.

    Temporary or ephemeral datasets are the ones created without specified name.
    They are useful for optimization purposes and should be automatically removed.

    Temp dataset has specific name format:
        "session_<name>_<session_uuid>_<dataset_uuid>"
    The <name> suffix is optional. Both <uuid>s are auto-generated.

    Temp dataset examples:
        session_myname_624b41_48e8b4
        session_4b962d_2a5dff

    Parameters:

    name (str): The name of the session. Only latters and numbers are supported.
           It can be empty.
    catalog (Catalog): Catalog object.
    """

    GLOBAL_SESSION_CTX: Optional["Session"] = None
    SESSION_CONTEXTS: ClassVar[list["Session"]] = []
    ORIGINAL_EXCEPT_HOOK = None

    DATASET_PREFIX = "session_"
    GLOBAL_SESSION_NAME = "global"
    SESSION_UUID_LEN = 6
    TEMP_TABLE_UUID_LEN = 6

    def __init__(
        self,
        name="",
        catalog: Optional["Catalog"] = None,
        client_config: Optional[dict] = None,
        in_memory: bool = False,
    ):
        if re.match(r"^[0-9a-zA-Z]*$", name) is None:
            raise ValueError(
                f"Session name can contain only letters or numbers - '{name}' given."
            )

        if not name:
            name = self.GLOBAL_SESSION_NAME

        session_uuid = uuid4().hex[: self.SESSION_UUID_LEN]
        self.name = f"{name}_{session_uuid}"
        self.is_new_catalog = not catalog
        self.catalog = catalog or get_catalog(
            client_config=client_config, in_memory=in_memory
        )
        self.dataset_versions: list[tuple[DatasetRecord, str, bool]] = []

        # Job management attributes
        self.job: Optional[Job] = None
        self.job_status: Optional[JobStatus] = None
        self.owns_job: Optional[bool] = None
        self._job_hooks_registered: bool = False
        self._job_finalize_hook: Optional[Callable[[], None]] = None

    def __enter__(self):
        # Push the current context onto the stack
        Session.SESSION_CONTEXTS.append(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._cleanup_created_versions()

        self._cleanup_temp_datasets()
        if self.is_new_catalog:
            self.catalog.metastore.close_on_exit()
            self.catalog.warehouse.close_on_exit()

        if Session.SESSION_CONTEXTS:
            Session.SESSION_CONTEXTS.pop()

    def add_dataset_version(
        self, dataset: "DatasetRecord", version: str, listing: bool = False
    ) -> None:
        self.dataset_versions.append((dataset, version, listing))

    def get_or_create_job(self) -> "Job":
        """
        Get or create a Job for this session.

        Returns:
            Job: The active Job instance.

        Behavior:
            - If a job already exists in this session, it is returned.
            - If ``DATACHAIN_JOB_ID`` is set, the corresponding job is fetched.
            - Otherwise, a new job is created:
                * Name = absolute path to the Python script.
                * Query = empty string.
                * Parent = last job with the same name, if available.
                * Status = "running".
              Exit hooks are registered to finalize the job.
        """
        if self.job:
            return self.job

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            # SaaS run: just fetch existing job
            self.job = self.catalog.metastore.get_job(env_job_id)
            if not self.job:
                raise JobNotFoundError(
                    f"Job {env_job_id} from DATACHAIN_JOB_ID env not found"
                )
            self.owns_job = False
        else:
            # Local run: create new job
            script = os.path.abspath(sys.argv[0]) if sys.argv else "interactive"
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

            # try to find the parent job
            parent = self.catalog.metastore.get_last_job_by_name(script)

            job_id = self.catalog.metastore.create_job(
                name=script,
                query="",
                query_type=JobQueryType.PYTHON,
                status=JobStatus.RUNNING,
                python_version=python_version,
                parent_job_id=parent.id if parent else None,
            )
            self.job = self.catalog.metastore.get_job(job_id)
            self.owns_job = True
            self.job_status = JobStatus.RUNNING

            # register cleanup hooks only once for the global session
            if not self._job_hooks_registered and self is self.GLOBAL_SESSION_CTX:

                def _finalize_success_hook() -> None:
                    self._finalize_job_success()

                self._job_finalize_hook = _finalize_success_hook
                atexit.register(self._job_finalize_hook)
                self._job_hooks_registered = True

        assert self.job is not None
        return self.job

    def reset_job_state(self):
        """
        Reset job state for testing purposes.
        Useful for simulating multiple script runs.
        """
        # Unregister atexit hook if registered
        if self._job_finalize_hook is not None:
            try:
                atexit.unregister(self._job_finalize_hook)
            except ValueError:
                # Hook was already unregistered
                pass
            self._job_finalize_hook = None

        # Clear job state
        self.job = None
        self.job_status = None
        self.owns_job = None
        self._job_hooks_registered = False

    def _finalize_job_success(self):
        """Mark the current job as completed."""
        if self.job and self.owns_job and self.job_status == JobStatus.RUNNING:
            self.catalog.metastore.set_job_status(self.job.id, JobStatus.COMPLETE)
            self.job_status = JobStatus.COMPLETE

    def _finalize_job_as_canceled(self):
        """Mark the current job as canceled."""
        if self.job and self.owns_job and self.job_status == JobStatus.RUNNING:
            self.catalog.metastore.set_job_status(self.job.id, JobStatus.CANCELED)
            self.job_status = JobStatus.CANCELED

    def _finalize_job_as_failed(self, exc_type, exc_value, tb):
        """Mark the current job as failed with error details."""
        if self.job and self.owns_job and self.job_status == JobStatus.RUNNING:
            error_stack = "".join(traceback.format_exception(exc_type, exc_value, tb))
            self.catalog.metastore.set_job_status(
                self.job.id,
                JobStatus.FAILED,
                error_message=str(exc_value),
                error_stack=error_stack,
            )
            self.job_status = JobStatus.FAILED

    def generate_temp_dataset_name(self) -> str:
        return self.get_temp_prefix() + uuid4().hex[: self.TEMP_TABLE_UUID_LEN]

    def get_temp_prefix(self) -> str:
        return f"{self.DATASET_PREFIX}{self.name}_"

    @classmethod
    def is_temp_dataset(cls, name) -> bool:
        return name.startswith(cls.DATASET_PREFIX)

    def _cleanup_temp_datasets(self) -> None:
        prefix = self.get_temp_prefix()
        try:
            for dataset in list(self.catalog.metastore.list_datasets_by_prefix(prefix)):
                self.catalog.remove_dataset(dataset.name, dataset.project, force=True)
        # suppress error when metastore has been reset during testing
        except TableMissingError:
            pass

    def _cleanup_created_versions(self) -> None:
        if not self.dataset_versions:
            return

        for dataset, version, listing in self.dataset_versions:
            if not listing:
                self.catalog.remove_dataset_version(dataset, version)

        self.dataset_versions.clear()

    @classmethod
    def get(
        cls,
        session: Optional["Session"] = None,
        catalog: Optional["Catalog"] = None,
        client_config: Optional[dict] = None,
        in_memory: bool = False,
    ) -> "Session":
        """Creates a Session() object from a catalog.

        Parameters:
            session (Session): Optional Session(). If not provided a new session will
                    be created. It's needed mostly for simple API purposes.
            catalog (Catalog): Optional catalog. By default, a new catalog is created.
        """
        if session:
            return session

        # Access the active (most recent) context from the stack
        if cls.SESSION_CONTEXTS:
            session = cls.SESSION_CONTEXTS[-1]

        elif cls.GLOBAL_SESSION_CTX is None:
            cls.GLOBAL_SESSION_CTX = Session(
                cls.GLOBAL_SESSION_NAME,
                catalog,
                client_config=client_config,
                in_memory=in_memory,
            )
            session = cls.GLOBAL_SESSION_CTX

            atexit.register(cls._global_cleanup)
            cls.ORIGINAL_EXCEPT_HOOK = sys.excepthook
            sys.excepthook = cls.except_hook
        else:
            session = cls.GLOBAL_SESSION_CTX

        if client_config and session.catalog.client_config != client_config:
            session = Session(
                "session" + uuid4().hex[:4],
                catalog,
                client_config=client_config,
                in_memory=in_memory,
            )
            session.__enter__()

        return session

    @staticmethod
    def except_hook(exc_type, exc_value, exc_traceback):
        # Handle KeyboardInterrupt specially - mark as canceled and exit with
        # signal code
        if exc_type is KeyboardInterrupt:
            if Session.GLOBAL_SESSION_CTX:
                Session.GLOBAL_SESSION_CTX._finalize_job_as_canceled()
                Session.GLOBAL_SESSION_CTX.__exit__(exc_type, exc_value, exc_traceback)
            Session._global_cleanup()

            if Session.ORIGINAL_EXCEPT_HOOK:
                Session.ORIGINAL_EXCEPT_HOOK(exc_type, exc_value, exc_traceback)

            # Exit with SIGINT signal code (128 + 2 = 130, or -2 in subprocess terms)
            sys.exit(130)
        else:
            # Regular exception - mark as failed
            if Session.GLOBAL_SESSION_CTX:
                Session.GLOBAL_SESSION_CTX._finalize_job_as_failed(
                    exc_type, exc_value, exc_traceback
                )
                Session.GLOBAL_SESSION_CTX.__exit__(exc_type, exc_value, exc_traceback)
            Session._global_cleanup()

            if Session.ORIGINAL_EXCEPT_HOOK:
                Session.ORIGINAL_EXCEPT_HOOK(exc_type, exc_value, exc_traceback)

    @classmethod
    def cleanup_for_tests(cls):
        if cls.GLOBAL_SESSION_CTX is not None:
            cls.GLOBAL_SESSION_CTX.__exit__(None, None, None)
            cls.GLOBAL_SESSION_CTX = None
            atexit.unregister(cls._global_cleanup)

        if cls.ORIGINAL_EXCEPT_HOOK:
            sys.excepthook = cls.ORIGINAL_EXCEPT_HOOK

    @staticmethod
    def _global_cleanup():
        if Session.GLOBAL_SESSION_CTX is not None:
            Session.GLOBAL_SESSION_CTX.__exit__(None, None, None)

        for obj in gc.get_objects():  # Get all tracked objects
            try:
                if isinstance(obj, Session):
                    # Cleanup temp dataset for session variables.
                    obj.__exit__(None, None, None)
            except ReferenceError:
                continue  # Object has been finalized already
            except Exception as e:  # noqa: BLE001
                logger.error(f"Exception while cleaning up session: {e}")  # noqa: G004
