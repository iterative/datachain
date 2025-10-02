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


@pytest.mark.parametrize(
    "exception_type,expected_status,should_have_error",
    [
        (RuntimeError("error"), JobStatus.FAILED, True),
        (KeyboardInterrupt(), JobStatus.CANCELED, False),
    ],
)
def test_finalize_failure(
    test_session,
    patch_argv,
    patch_user_script,
    exception_type,
    expected_status,
    should_have_error,
):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    # Mock sys.exit in the job module for KeyboardInterrupt to avoid actually exiting
    with patch("datachain.job.sys.exit") as mock_exit:
        try:
            raise exception_type
        except (RuntimeError, KeyboardInterrupt) as e:
            jm.finalize_failure(test_session, type(e), e, e.__traceback__)

        # Verify exit code for KeyboardInterrupt
        if isinstance(exception_type, KeyboardInterrupt):
            # Check that sys.exit was called with 130 (may be called multiple
            # times due to hook chaining)
            assert mock_exit.called
            assert all(call[0][0] == 130 for call in mock_exit.call_args_list)

    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == expected_status

    if should_have_error:
        assert "error" in db_job.error_message
        assert "RuntimeError" in db_job.error_stack
    else:
        # KeyboardInterrupt should not set error message/stack
        assert db_job.error_message == ""
        assert db_job.error_stack == ""


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


def test_finalize_failure_delegates_to_previous_excepthook(
    test_session, patch_argv, patch_user_script
):
    jm = JobManager()
    job = jm.get_or_create(test_session)

    called = {}

    def fake_excepthook(exc_type, exc_value, tb):
        called["exc"] = (exc_type, str(exc_value))

    # Replace the previous hook that JobManager saved
    old_hook = jm._previous_excepthook
    jm._previous_excepthook = fake_excepthook

    try:
        try:
            raise ValueError("bad stuff")
        except ValueError as e:
            jm.finalize_failure(test_session, type(e), e, e.__traceback__)
    finally:
        jm._previous_excepthook = old_hook

    assert "bad stuff" in called["exc"][1]
    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job.status == JobStatus.FAILED


def test_reset_clears_state(test_session, patch_argv, patch_user_script):
    """Test that reset() clears JobManager state."""
    jm = JobManager()
    jm.get_or_create(test_session)

    # Verify initial state
    assert jm.job is not None
    assert jm.status == JobStatus.RUNNING
    assert jm.owned is True
    assert jm._hooks_registered is True
    assert len(jm._hook_refs) > 0

    # Reset
    jm.reset()

    # Verify state is cleared
    assert jm.job is None
    assert jm.status is None
    assert jm.owned is None
    assert jm._hooks_registered is False
    assert len(jm._hook_refs) == 0


def test_reset_restores_excepthook(test_session, patch_argv, patch_user_script):
    """Test that reset() restores sys.excepthook."""
    original_hook = sys.excepthook
    jm = JobManager()
    jm.get_or_create(test_session)

    # Hook should be changed
    assert sys.excepthook != original_hook

    # Reset should restore it to what it was before JobManager modified it
    jm.reset()
    assert sys.excepthook == original_hook


def test_reset_allows_recreation(test_session, patch_argv, patch_user_script):
    """Test that after reset(), get_or_create() works again."""
    jm = JobManager()
    job1 = jm.get_or_create(test_session)
    job1_id = job1.id

    jm.reset()

    # Should be able to create a new job
    job2 = jm.get_or_create(test_session)
    assert job2.id != job1_id
    assert jm.job is job2
    assert jm.status == JobStatus.RUNNING
