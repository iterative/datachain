import os
from typing import Optional, Union

metrics: dict[str, Union[str, int, float, bool, None]] = {}


def set(key: str, value: Union[str, int, float, bool, None]) -> None:  # noqa: PYI041
    """Set a metric value."""
    if not isinstance(key, str):
        raise TypeError("Key must be a string")
    if not key:
        raise ValueError("Key must not be empty")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("Value must be a string, int, float or bool")
    metrics[key] = value

    if job_id := os.getenv("DATACHAIN_JOB_ID"):
        from datachain.query.session import Session

        metastore = Session.get().catalog.metastore
        metastore.update_job(job_id, metrics=metrics)


def get(key: str) -> Optional[Union[str, int, float, bool]]:
    """Get a metric value."""
    return metrics[key]
