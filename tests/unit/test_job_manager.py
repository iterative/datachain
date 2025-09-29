import sys
from unittest.mock import patch

import pytest

from datachain.data_storage import JobStatus
from datachain.job import JobManager


@pytest.fixture
def patch_argv(monkeypatch, tmp_path):
    """Patch sys.argv to always point to the same fake script."""
    fake_script = tmp_path / "script.py"
    fake_script.write_text("print('hello world')\n")
    monkeypatch.setattr("sys.argv", [str(fake_script)])
    yield


@pytest.fixture
def patch_user_script():
    """Patch get_user_script_source to always return deterministic code."""
    with patch(
        "datachain.job.get_user_script_source", return_value="print('hello world')\n"
    ):
        yield


def test_reuse_job_from_env_var(test_session, monkeypatch):
    """If DATACHAIN_JOB_ID is set, JobManager should reuse that job."""
    job_id = test_session.catalog.metastore.create_job(
        "my-job",
        "print('hello world')\n",
    )

    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    jm = JobManager()
    job = jm.get_or_create(test_session)

    assert job.id == job_id
    assert job.name == "my-job"
    assert jm.owned is False
    assert jm.status is None


def test_get_or_create_creates_job(test_session, patch_argv, patch_user_script):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job is not None
    assert db_job.name.endswith("script.py")
    assert db_job.query == "print('hello world')\n"
    assert db_job.status == JobStatus.RUNNING


def test_finalize_success(test_session, patch_argv, patch_user_script):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    jm.finalize_success(test_session)

    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.COMPLETE
    assert db_job.finished_at is not None


def test_finalize_failure(test_session, patch_argv, patch_user_script):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    try:
        raise RuntimeError("error")
    except RuntimeError as e:
        jm.finalize_failure(test_session, type(e), e, e.__traceback__)

    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.FAILED
    assert "error" in db_job.error_message
    assert "RuntimeError" in db_job.error_stack


def test_get_or_create_is_idempotent(test_session, patch_argv, patch_user_script):
    jm = JobManager()
    job1 = jm.get_or_create(test_session)
    job2 = jm.get_or_create(test_session)

    assert job1 is job2
    assert jm.job is job1


def test_get_or_create_links_to_parent(test_session, patch_argv, patch_user_script):
    jm1 = JobManager()
    job1 = jm1.get_or_create(test_session)
    jm1.finalize_success(test_session)

    jm2 = JobManager()
    job2 = jm2.get_or_create(test_session)

    assert job2.parent_job_id == job1.id


def test_get_or_create_fallback_query(test_session, patch_argv):
    # Patch to return None for script source
    with patch("datachain.job.get_user_script_source", return_value=None):
        jm = JobManager()
        job = jm.get_or_create(test_session)

    assert job.query.startswith("python ")
    assert job.name.endswith("script.py")


def test_finalize_failure_delegates_to_sys_excepthook(
    test_session, patch_argv, patch_user_script
):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    called = {}

    def fake_excepthook(exc_type, exc_value, tb):
        called["exc"] = (exc_type, str(exc_value))

    sys.__excepthook__, old_hook = fake_excepthook, sys.__excepthook__

    try:
        try:
            raise ValueError("bad stuff")
        except ValueError as e:
            jm.finalize_failure(test_session, type(e), e, e.__traceback__)
    finally:
        sys.__excepthook__ = old_hook

    assert "bad stuff" in called["exc"][1]
    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.FAILED
