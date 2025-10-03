from unittest.mock import patch

import pytest

from datachain.data_storage import JobStatus
from datachain.query.session import Session


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
        "datachain.utils.get_user_script_source", return_value="print('hello world')\n"
    ):
        yield


def test_reuse_job_from_env_var(test_session, monkeypatch):
    """If DATACHAIN_JOB_ID is set, Session should reuse that job."""
    job_id = test_session.catalog.metastore.create_job(
        "my-job",
        "print('hello world')\n",
    )

    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    job = test_session.get_or_create_job()

    assert job.id == job_id
    assert job.name == "my-job"
    assert test_session.owns_job is False
    assert test_session.job_status is None


def test_get_or_create_creates_job(test_session, patch_argv, patch_user_script):
    job = test_session.get_or_create_job()

    db_job = test_session.catalog.metastore.get_job(job.id)
    assert db_job is not None
    assert db_job.name.endswith("script.py")
    assert db_job.query == "print('hello world')\n"
    assert db_job.status == JobStatus.RUNNING


def test_finalize_success(test_session, patch_argv, patch_user_script):
    job = test_session.get_or_create_job()

    test_session._finalize_job_success()

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
    job = test_session.get_or_create_job()

    try:
        raise exception_type
    except (RuntimeError, KeyboardInterrupt) as e:
        if isinstance(exception_type, KeyboardInterrupt):
            test_session._finalize_job_as_canceled()
        else:
            test_session._finalize_job_as_failed(type(e), e, e.__traceback__)

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
    job1 = test_session.get_or_create_job()
    job2 = test_session.get_or_create_job()

    assert job1 is job2
    assert test_session.job is job1


def test_get_or_create_links_to_parent(test_session, patch_argv, patch_user_script):
    job1 = test_session.get_or_create_job()
    test_session._finalize_job_success()

    # Create a new session to get a new job
    session2 = Session(catalog=test_session.catalog)
    job2 = session2.get_or_create_job()

    assert job2.parent_job_id == job1.id


def test_get_or_create_fallback_query(test_session, patch_argv):
    # Patch to return None for script source
    with patch("datachain.utils.get_user_script_source", return_value=None):
        job = test_session.get_or_create_job()

    assert job.query.startswith("python ")
    assert job.name.endswith("script.py")


def test_except_hook_delegates_to_original(test_session, patch_argv, patch_user_script):
    """Test that Session.except_hook delegates to ORIGINAL_EXCEPT_HOOK."""
    # Set up global session for this test
    Session.GLOBAL_SESSION_CTX = test_session
    test_session.get_or_create_job()

    called = {}

    def fake_excepthook(exc_type, exc_value, tb):
        called["exc"] = (exc_type, str(exc_value))

    # Save and replace the original hook
    old_hook = Session.ORIGINAL_EXCEPT_HOOK
    Session.ORIGINAL_EXCEPT_HOOK = fake_excepthook

    try:
        try:
            raise ValueError("bad stuff")
        except ValueError as e:
            Session.except_hook(type(e), e, e.__traceback__)
    finally:
        Session.ORIGINAL_EXCEPT_HOOK = old_hook
        Session.GLOBAL_SESSION_CTX = None

    assert "bad stuff" in called["exc"][1]
    db_job = test_session.catalog.metastore.get_job(test_session.job.id)
    assert db_job.status == JobStatus.FAILED
