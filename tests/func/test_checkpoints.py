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
    assert len(checkpoints_before) == 4
    assert all(c.partial is False for c in checkpoints_before)

    # Verify UDF tables exist by checking all tables with udf_ prefix
    # Note: Due to checkpoint skipping, some jobs may reuse parent tables
    all_udf_tables_before = warehouse.db.list_tables(prefix="udf_")

    # At least some UDF tables should exist from the operations
    assert len(all_udf_tables_before) > 0

    # Modify checkpoint created_at to be older than TTL (4 hours by default)
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for checkpoint in checkpoints_before:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Run cleanup_checkpoints with default TTL (4 hours)
    catalog.cleanup_checkpoints()

    # Verify checkpoints were removed
    checkpoints_after = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_after) == 0

    # Verify job-specific UDF tables were removed
    job_id_sanitized = job_id.replace("-", "")
    udf_tables_after = warehouse.db.list_tables(prefix=f"udf_{job_id_sanitized}_")
    assert len(udf_tables_after) == 0


def test_cleanup_checkpoints_with_custom_ttl(test_session, monkeypatch, nums_dataset):
    """Test that cleanup_checkpoints respects custom TTL parameter."""
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create some checkpoints
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints) == 2
    assert all(c.partial is False for c in checkpoints)

    # Modify all checkpoints to be 2 hours old
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    for checkpoint in checkpoints:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Run cleanup with custom TTL of 1 hour (3600 seconds)
    # Checkpoints are 2 hours old, so they should be removed
    catalog.cleanup_checkpoints(ttl_seconds=3600)

    # Verify checkpoints were removed
    assert len(list(metastore.list_checkpoints(job_id))) == 0


def test_clean_job_checkpoints(test_session, monkeypatch, nums_dataset):
    """Test that clean_job_checkpoints removes all checkpoints for a specific job."""
    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create checkpoints for two different jobs
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    first_job_id = test_session.get_or_create_job().id
    first_job = metastore.get_job(first_job_id)

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    second_job_id = test_session.get_or_create_job().id

    # Verify both jobs have checkpoints
    first_checkpoints = list(metastore.list_checkpoints(first_job_id))
    second_checkpoints = list(metastore.list_checkpoints(second_job_id))
    assert len(first_checkpoints) == 2
    assert len(second_checkpoints) == 2

    # Clean up only first job's checkpoints using clean_job_checkpoints
    catalog.clean_job_checkpoints(first_job)

    # Verify only first job's checkpoints were removed
    assert len(list(metastore.list_checkpoints(first_job_id))) == 0
    assert len(list(metastore.list_checkpoints(second_job_id))) == 2


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
    assert len(checkpoints_before) == 2

    # Run cleanup (should not remove recent checkpoints)
    catalog.cleanup_checkpoints()

    # Verify checkpoints were not removed
    checkpoints_after = list(metastore.list_checkpoints(job_id))
    assert len(checkpoints_after) == 2
    checkpoint_ids_before = {cp.id for cp in checkpoints_before}
    checkpoint_ids_after = {cp.id for cp in checkpoints_after}
    assert checkpoint_ids_before == checkpoint_ids_after


def test_cleanup_checkpoints_preserves_with_active_descendants(
    test_session, nums_dataset
):
    """
    Test that outdated parent checkpoints are preserved when descendants have
    active checkpoints.
    """
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create parent job with checkpoints
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    parent_job_id = test_session.get_or_create_job().id

    # Create child job (will have parent_job_id set) with more recent checkpoints
    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    child_job_id = test_session.get_or_create_job().id

    # Verify parent job is set correctly
    child_job = metastore.get_job(child_job_id)
    assert child_job.parent_job_id == parent_job_id

    # Make parent checkpoints old (outdated)
    parent_checkpoints = list(metastore.list_checkpoints(parent_job_id))
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for checkpoint in parent_checkpoints:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Child checkpoints remain recent (within TTL)
    child_checkpoints = list(metastore.list_checkpoints(child_job_id))
    assert len(child_checkpoints) > 0

    # Run cleanup with default TTL (4 hours)
    catalog.cleanup_checkpoints()

    # Verify parent checkpoints were NOT removed (child still needs them)
    parent_after = list(metastore.list_checkpoints(parent_job_id))
    assert len(parent_after) == len(parent_checkpoints)

    # Child checkpoints should still be there
    child_after = list(metastore.list_checkpoints(child_job_id))
    assert len(child_after) == len(child_checkpoints)


def test_cleanup_checkpoints_partial_job_cleanup(test_session, nums_dataset):
    """Test that only outdated checkpoints are removed, not all checkpoints in a job."""
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create a job with multiple checkpoints at different times
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)

    # First checkpoint
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    job_id = test_session.get_or_create_job().id

    first_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(first_checkpoints) == 2

    # Make first checkpoints old (outdated)
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)
    for checkpoint in first_checkpoints:
        metastore.db.execute(
            metastore._checkpoints.update()
            .where(ch.c.id == checkpoint.id)
            .values(created_at=old_time)
        )

    # Create more checkpoints in the same job (recent, within TTL)
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")

    all_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(all_checkpoints) == 4  # 2 old + 2 new

    # Run cleanup with default TTL (4 hours)
    catalog.cleanup_checkpoints()

    # Verify only outdated checkpoints were removed
    remaining_checkpoints = list(metastore.list_checkpoints(job_id))
    assert len(remaining_checkpoints) == 2  # Only recent ones remain

    # Verify the remaining are the new ones (not in first_checkpoints)
    first_ids = {cp.id for cp in first_checkpoints}
    remaining_ids = {cp.id for cp in remaining_checkpoints}
    assert first_ids.isdisjoint(remaining_ids), "Old checkpoints should be gone"


def test_cleanup_checkpoints_branch_pruning(test_session, nums_dataset):
    """
    Test that entire outdated job lineages are cleaned in one pass (branch pruning).
    """
    from datetime import datetime, timedelta, timezone

    catalog = test_session.catalog
    metastore = catalog.metastore

    # Create a lineage: root -> child -> grandchild
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session)
    chain.map(doubled=lambda num: num * 2, output=int).save("nums_doubled")
    root_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(tripled=lambda num: num * 3, output=int).save("nums_tripled")
    child_job_id = test_session.get_or_create_job().id

    reset_session_job_state()
    chain.map(quadrupled=lambda num: num * 4, output=int).save("nums_quadrupled")
    grandchild_job_id = test_session.get_or_create_job().id

    # Verify lineage
    child_job = metastore.get_job(child_job_id)
    grandchild_job = metastore.get_job(grandchild_job_id)
    assert child_job.parent_job_id == root_job_id
    assert grandchild_job.parent_job_id == child_job_id

    # Make ALL checkpoints outdated (older than TTL)
    all_job_ids = [root_job_id, child_job_id, grandchild_job_id]
    ch = metastore._checkpoints
    old_time = datetime.now(timezone.utc) - timedelta(hours=5)

    for job_id in all_job_ids:
        checkpoints = list(metastore.list_checkpoints(job_id))
        for checkpoint in checkpoints:
            metastore.db.execute(
                metastore._checkpoints.update()
                .where(ch.c.id == checkpoint.id)
                .values(created_at=old_time)
            )

    # Run cleanup once
    catalog.cleanup_checkpoints()

    # Verify ALL jobs were cleaned in single pass (branch pruning)
    for job_id in all_job_ids:
        remaining = list(metastore.list_checkpoints(job_id))
        assert len(remaining) == 0, f"Job {job_id} should have been cleaned"


def test_udf_generator_continue_parallel(test_session_tmpfile, monkeypatch):
    """Test continuing RowGenerator from partial with parallel=True.

    This tests that processed table is properly passed through parallel
    execution path so that checkpoint recovery works correctly.
    """
    from datachain.query.dataset import UDFStep

    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    # Track which numbers have been processed
    processed_nums = []
    run_count = {"count": 0}

    class GenMultiple(dc.Generator):
        """Generator that yields multiple outputs per input."""

        def process(self, num):
            processed_nums.append(num)
            # Fail on input 4 in first run only
            if num == 4 and run_count["count"] == 0:
                raise Exception(f"Simulated failure on num={num}")
            # Each input yields 2 outputs
            yield num * 10
            yield num

    dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")

    # -------------- FIRST RUN (FAILS) -------------------
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(parallel=2, batch_size=2)
        .gen(result=GenMultiple(), output=int)
    )

    with pytest.raises(RuntimeError):
        chain.save("results")

    first_job_id = test_session.get_or_create_job().id
    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 1
    hash_input = checkpoints[0].hash

    # Verify partial output table exists
    partial_table_name = UDFStep.partial_output_table_name(first_job_id, hash_input)
    assert warehouse.db.has_table(partial_table_name)

    # Verify processed table exists and has tracked some inputs
    processed_table_name = UDFStep.processed_table_name(first_job_id, hash_input)
    assert warehouse.db.has_table(processed_table_name)
    processed_table = warehouse.get_table(processed_table_name)
    processed_count_first = warehouse.table_rows_count(processed_table)
    assert processed_count_first > 0, "Some inputs should be tracked"

    # -------------- SECOND RUN (CONTINUE) -------------------
    reset_session_job_state()

    # Clear processed list and increment run count
    processed_nums.clear()
    run_count["count"] += 1

    # Should complete successfully
    chain.save("results")

    # Verify result
    result = (
        dc.read_dataset("results", session=test_session)
        .order_by("result")
        .to_list("result")
    )
    # Each of 6 inputs yields 2 outputs: [10,1], [20,2], ..., [60,6]
    assert result == [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (10,),
        (20,),
        (30,),
        (40,),
        (50,),
        (60,),
    ]

    # Verify only unprocessed inputs were processed in second run
    # (should be less than all 6 inputs)
    assert len(processed_nums) < 6
