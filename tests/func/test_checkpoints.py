import pytest

import datachain as dc
from datachain.error import DatasetNotFoundError
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


def test_checkpoints_parallel(test_session_tmpfile, monkeypatch):
    def mapper_fail(num) -> int:
        raise Exception("Error")

    test_session = test_session_tmpfile
    catalog = test_session.catalog

    dc.read_values(num=list(range(1000)), session=test_session).save("nums")

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    chain = dc.read_dataset("nums", session=test_session).settings(parallel=True)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(RuntimeError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = test_session.get_or_create_job().id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    assert len(catalog.get_dataset("nums1").versions) == 1
    assert len(catalog.get_dataset("nums2").versions) == 1
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


def test_cleanup_checkpoints_with_ttl(test_session, monkeypatch, nums_dataset):
    """Test that cleanup_checkpoints removes old checkpoints and their UDF tables."""
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    # Create some checkpoints by running a chain with map (which creates UDF tables)
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    job_id = test_session.get_or_create_job().id

    checkpoints_before = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_before) == 6

    # Verify UDF tables exist
    # Tables are now shared (no job_id) and named udf_{hash}_input and udf_{hash}_output
    udf_tables = []
    for checkpoint in checkpoints_before:
        table_prefix = f"udf_{checkpoint.hash}"
        matching_tables = warehouse.db.list_tables(prefix=table_prefix)
        udf_tables.extend(matching_tables)

    # At least some UDF tables should exist
    assert len(udf_tables) > 0

    # Modify checkpoint created_at to be older than TTL (4 hours by default)
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for checkpoint in checkpoints_before:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Run cleanup_checkpoints
    catalog.cleanup_checkpoints()

    # Verify checkpoints were removed
    checkpoints_after = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_after) == 0

    # Verify UDF tables were removed
    for table_name in udf_tables:
        assert not warehouse.db.has_table(table_name)


def test_cleanup_checkpoints_with_custom_ttl(test_session, monkeypatch, nums_dataset):
    """Test that cleanup_checkpoints respects custom TTL from environment variable."""
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Set custom TTL to 1 hour
    monkeypatch.setenv("CHECKPOINT_TTL", "3600")

    # Create some checkpoints
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints) == 3

    # Modify all checkpoints to be 2 hours old (older than custom TTL)
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    for checkpoint in checkpoints:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Run cleanup with custom TTL
    catalog.cleanup_checkpoints()

    # Verify checkpoints were removed
    assert len(list(metastore.list_checkpoints(job_id))) == 0


def test_cleanup_checkpoints_for_specific_job(test_session, monkeypatch, nums_dataset):
    """Test that cleanup_checkpoints can target a specific job."""
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create checkpoints for two different jobs
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    first_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    second_job_id = test_session.get_or_create_job().id

    # Verify both jobs have checkpoints
    first_checkpoints = list(metastore.list_checkpoints(first_job_id))
    second_checkpoints = list(metastore.list_checkpoints(second_job_id))
    assert len(first_checkpoints) == 3
    assert len(second_checkpoints) == 3

    # Make both checkpoints old
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for checkpoint in first_checkpoints + second_checkpoints:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Clean up only first job's checkpoints
    catalog.cleanup_checkpoints(job_id=first_job_id)

    # Verify only first job's checkpoints were removed
    assert len(list(metastore.list_checkpoints(first_job_id))) == 0
    assert len(list(metastore.list_checkpoints(second_job_id))) == 3


def test_cleanup_checkpoints_no_old_checkpoints(test_session, nums_dataset):
    """Test that cleanup_checkpoints does nothing when no old checkpoints exist."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create a recent checkpoint
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints_before = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_before) == 3

    # Run cleanup (should not remove recent checkpoints)
    catalog.cleanup_checkpoints()

    # Verify checkpoints were not removed
    checkpoints_after = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_after) == 3
    checkpoint_ids_before = {cp.id for cp in checkpoints_before}
    checkpoint_ids_after = {cp.id for cp in checkpoints_after}
    assert checkpoint_ids_before == checkpoint_ids_after


def test_cleanup_checkpoints_created_after(test_session, nums_dataset):
    """Test that cleanup_checkpoints can invalidate checkpoints after a certain time."""
    import time
    from datetime import datetime, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore
    warehouse = catalog.warehouse

    # Create first checkpoint
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    # Get the first set of checkpoints
    first_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(first_checkpoints) == 3

    # Sleep a tiny bit to ensure different timestamps
    time.sleep(0.01)

    # Record the cutoff time
    cutoff_time = datetime.now(timezone.utc)

    # Sleep again to ensure next checkpoints are after cutoff
    time.sleep(0.01)

    # Create second checkpoint (simulating re-run with code changes)
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")

    # Verify we now have more checkpoints
    all_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(all_checkpoints) == 6

    # Get UDF tables before cleanup
    # Tables are now shared (no job_id), so just count all UDF tables
    all_udf_tables_before = warehouse.db.list_tables(prefix="udf_")
    assert len(all_udf_tables_before) > 0

    # Clean up checkpoints created after the cutoff time
    catalog.cleanup_checkpoints(job_id=job_id, created_after=cutoff_time)

    # Verify only first checkpoints remain
    remaining_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(remaining_checkpoints) == 3

    # Verify the remaining checkpoints are the first ones
    remaining_ids = {cp.id for cp in remaining_checkpoints}
    first_ids = {cp.id for cp in first_checkpoints}
    assert remaining_ids == first_ids

    # Verify UDF tables for removed checkpoints are gone
    all_udf_tables_after = warehouse.db.list_tables(prefix=f"udf_{job_id}_")
    # Should have fewer tables now
    assert len(all_udf_tables_after) < len(all_udf_tables_before)


def test_cleanup_checkpoints_created_after_with_multiple_jobs(
    test_session, nums_dataset
):
    """Test created_after with specific job_id doesn't affect other jobs."""
    import time
    from datetime import datetime, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create checkpoints for first job
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    first_job_id = test_session.get_or_create_job().id

    time.sleep(0.01)
    cutoff_time = datetime.now(timezone.utc)
    time.sleep(0.01)

    # Create more checkpoints for first job
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")

    # Create checkpoints for second job (after cutoff)
    reset_session_job_state()
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    second_job_id = test_session.get_or_create_job().id

    # Verify initial state
    first_job_checkpoints = list(metastore.list_checkpoints(first_job_id))
    second_job_checkpoints = list(metastore.list_checkpoints(second_job_id))
    assert len(first_job_checkpoints) == 6
    assert len(second_job_checkpoints) == 3

    # Clean up only first job's checkpoints created after cutoff
    catalog.cleanup_checkpoints(job_id=first_job_id, created_after=cutoff_time)

    first_job_after = list(metastore.list_checkpoints(first_job_id))
    assert len(first_job_after) == 3

    second_job_after = list(metastore.list_checkpoints(second_job_id))
    assert len(second_job_after) == 3
