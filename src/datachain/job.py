import atexit
import json
import os
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union

from datachain.data_storage import JobQueryType, JobStatus
from datachain.error import JobNotFoundError
from datachain.utils import get_user_script_source

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
    parent_job_id: Optional[str] = None

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
        parent_job_id: Optional[str],
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
            parent_job_id,
        )


class JobManager:
    """
    Manages the lifecycle of a DataChain Job for a single Python process.

    Behavior:
      - If the environment variable ``DATACHAIN_JOB_ID`` is set (SaaS mode),
        the JobManager attaches to that job and does not manage its lifecycle.
      - If ``DATACHAIN_JOB_ID`` is not set (local script run), the JobManager
        will:
          * Create a new job before any work is done.
          * Use the script path as the job name.
          * Store the script source (if available) as the job query.
          * Link to the most recent job with the same name as its parent, if one exists.
          * Automatically mark the job as ``COMPLETE`` on normal exit, or ``FAILED`` if
            an unhandled exception terminates the process.
    """

    _hook_refs: ClassVar[list[Callable]] = []

    def __init__(self):
        self.job = None
        self.status = None
        self.owned = None  # True if this manager owns the Job lifecycle
        self._hooks_registered = False

    def get_or_create(self, session):
        """
        Return the active Job for this process, creating it if needed.

        Args:
            session (Session): The current DataChain session.

        Returns:
            Job: The active Job instance.

        Behavior:
            - If a job already exists in this JobManager, it is returned.
            - If ``DATACHAIN_JOB_ID`` is set, the corresponding job is fetched.
            - Otherwise, a new job is created:
                * Name = absolute path to the Python script.
                * Query = script source code if available, otherwise the command line.
                * Parent = last job with the same name, if available.
                * Status = "running".
              Exit hooks are registered to finalize the job.
        """

        if self.job:
            return self.job

        if env_job_id := os.getenv("DATACHAIN_JOB_ID"):
            # SaaS run: just fetch existing job
            self.job = session.catalog.metastore.get_job(env_job_id)
            if not self.job:
                raise JobNotFoundError(
                    f"Job {env_job_id} from DATACHAIN_JOB_ID env not found"
                )
            self.owned = False
        else:
            # Local run: create new job
            script = os.path.abspath(sys.argv[0]) if sys.argv else "interactive"
            source_code = get_user_script_source()
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

            # try to find the parent job
            parent = session.catalog.metastore.get_last_job_by_name(script)

            job_id = session.catalog.metastore.create_job(
                name=script,
                query=source_code or f"python {script}",
                query_type=JobQueryType.PYTHON,
                status=JobStatus.RUNNING,
                python_version=python_version,
                parent_job_id=parent.id if parent else None,
            )
            self.job = session.catalog.metastore.get_job(job_id)
            self.owned = True
            self.status = JobStatus.RUNNING

            # register cleanup hooks
            if not self._hooks_registered:
                # Register and remember hook
                def _finalize_success_hook() -> None:
                    self.finalize_success(session)

                atexit.register(_finalize_success_hook)
                self._hook_refs.append(_finalize_success_hook)

                sys.excepthook = lambda et, ev, tb: self.finalize_failure(
                    session, et, ev, tb
                )
                self._hooks_registered = True

        return self.job

    def finalize_success(self, session):
        """
        Mark the current job as completed.

        This is called automatically at process exit if no unhandled exception occurs,
        but can also be called manually.

        Args:
            session (Session): The current DataChain session.
        """
        if self.job and self.owned and self.status == JobStatus.RUNNING:
            session.catalog.metastore.set_job_status(self.job.id, JobStatus.COMPLETE)
            self.status = JobStatus.COMPLETE

    def finalize_failure(self, session, exc_type, exc_value, tb):
        """
        Mark the current job as failed.

        This is called automatically by sys.excepthook if an unhandled exception occurs.

        Args:
            session (Session): The current DataChain session.
            exc_type (type): Exception class.
            exc_value (Exception): Exception instance.
            tb (traceback): Traceback object.
        """
        if self.job and self.owned and self.status == JobStatus.RUNNING:
            error_stack = "".join(traceback.format_exception(exc_type, exc_value, tb))
            session.catalog.metastore.set_job_status(
                self.job.id,
                JobStatus.FAILED,
                error_message=str(exc_value),
                error_stack=error_stack,
            )
            self.status = JobStatus.FAILED
        # Delegate to default handler so exception still prints
        sys.__excepthook__(exc_type, exc_value, tb)


job_manager = JobManager()
