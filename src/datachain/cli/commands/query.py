import os
import sys
import traceback
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def query(
    catalog: "Catalog",
    script: str,
    parallel: Optional[int] = None,
    params: Optional[dict[str, str]] = None,
) -> None:
    from datachain.data_storage import JobQueryType, JobStatus

    with open(script, encoding="utf-8") as f:
        script_content = f.read()

    if parallel is not None:
        # This also sets this environment variable for any subprocesses
        os.environ["DATACHAIN_SETTINGS_PARALLEL"] = str(parallel)

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_executable = sys.executable

    job_id = catalog.metastore.create_job(
        name=os.path.basename(script),
        query=script_content,
        query_type=JobQueryType.PYTHON,
        python_version=python_version,
        params=params,
    )

    try:
        catalog.query(
            script_content,
            python_executable=python_executable,
            params=params,
            job_id=job_id,
        )
    except Exception as e:
        error_message = str(e)
        error_stack = traceback.format_exc()
        catalog.metastore.set_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_message,
            error_stack=error_stack,
        )
        raise
    catalog.metastore.set_job_status(job_id, JobStatus.COMPLETE)
