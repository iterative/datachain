import pytest
import sqlalchemy as sa

import datachain as dc
from datachain.error import DatasetNotFoundError, JobNotFoundError
from datachain.lib.utils import DataChainError
from datachain.query.dataset import UDFStep
from tests.utils import reset_session_job_state


def mapper_fail(num) -> int:
    raise Exception("Error")


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3, 4, 5, 6], session=test_session).save("nums")


@pytest.mark.parametrize("reset_checkpoints", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("use_datachain_job_id_env", [True, False])
def test_checkpoints(
    test_session,
    monkeypatch,
    nums_dataset,
    reset_checkpoints,
    with_delta,
    use_datachain_job_id_env,
):
    catalog = test_session.catalog
    metastore = catalog.metastore

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    if with_delta:
        chain = dc.read_dataset(
            "nums", delta=True, delta_on=["num"], session=test_session
        )
    else:
        chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID", metastore.create_job("my-job", "echo 1;")
        )

    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(DataChainError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = test_session.get_or_create_job().id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID",
            metastore.create_job("my-job", "echo 1;", parent_job_id=first_job_id),
        )
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    expected_versions = 1 if with_delta or not reset_checkpoints else 2
    assert len(catalog.get_dataset("nums1").versions) == expected_versions
    assert len(catalog.get_dataset("nums2").versions) == expected_versions
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    first_job_id = test_session.get_or_create_job().id

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")  # added change from first run
    chain.save("nums3")
    second_job_id = test_session.get_or_create_job().id

    assert len(catalog.get_dataset("nums1").versions) == 2 if reset_checkpoints else 1
    assert len(catalog.get_dataset("nums2").versions) == 2
    assert len(catalog.get_dataset("nums3").versions) == 2

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(DataChainError):
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

    # -------------- THIRD RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    with pytest.raises(DataChainError):
        chain.map(new=mapper_fail).save("nums3")
    third_job_id = test_session.get_or_create_job().id

    # -------------- FOURTH RUN -------------------
    reset_session_job_state()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    chain.save("nums3")
    fourth_job_id = test_session.get_or_create_job().id

    num1_versions = len(catalog.get_dataset("nums1").versions)
    num2_versions = len(catalog.get_dataset("nums2").versions)
    num3_versions = len(catalog.get_dataset("nums3").versions)

    if reset_checkpoints:
        assert num1_versions == 4
        assert num2_versions == 4
        assert num3_versions == 2

    else:
        assert num1_versions == 1
        assert num2_versions == 2
        assert num3_versions == 2

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(fourth_job_id))) == 3


def test_checkpoints_check_valid_chain_is_returned(
    test_session,
    monkeypatch,
    nums_dataset,
):
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.save("nums1")

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    ds = chain.save("nums1")

    # checking that we return expected DataChain even though we skipped chain creation
    # because of the checkpoints
    assert ds.dataset is not None
    assert ds.dataset.name == "nums1"
    assert len(ds.dataset.versions) == 1
    assert ds.order_by("num").to_list("num") == [(1,), (2,), (3,), (4,), (5,), (6,)]


def test_checkpoints_invalid_parent_job_id(test_session, monkeypatch, nums_dataset):
    # setting wrong job id
    reset_session_job_state()
    monkeypatch.setenv("DATACHAIN_JOB_ID", "caee6c54-6328-4bcd-8ca6-2b31cb4fff94")
    with pytest.raises(JobNotFoundError):
        dc.read_dataset("nums", session=test_session).save("nums1")


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_udf_checkpoints_cross_job_reuse(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(reset_checkpoints))

    # Track how many times the mapper is called
    call_count = {"count": 0}

    def double_num(num) -> int:
        call_count["count"] += 1
        return num * 2

    chain = dc.read_dataset("nums", session=test_session).map(
        doubled=double_num, output=int
    )

    # -------------- FIRST RUN - count() triggers UDF execution -------------------
    reset_session_job_state()
    assert chain.count() == 6
    first_job_id = test_session.get_or_create_job().id

    assert call_count["count"] == 6

    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 2, "Should have 2 checkpoints (before and after UDF)"

    # -------------- SECOND RUN - should reuse UDF checkpoint -------------------
    reset_session_job_state()
    call_count["count"] = 0  # Reset counter

    assert chain.count() == 6
    second_job_id = test_session.get_or_create_job().id

    if reset_checkpoints:
        assert call_count["count"] == 6, "Mapper should be called again"
    else:
        assert call_count["count"] == 0, "Mapper should NOT be called"

    # Check that second job created checkpoints
    checkpoints_second = list(catalog.metastore.list_checkpoints(second_job_id))
    if reset_checkpoints:
        # With reset, both checkpoints are created (hash_before and hash_after)
        assert len(checkpoints_second) == 2
    else:
        # Without reset, only hash_after checkpoint is created when skipping
        assert len(checkpoints_second) == 1

    # Verify the data is correct
    result = chain.order_by("num").to_list("doubled")
    assert result == [(2,), (4,), (6,), (8,), (10,), (12,)]


def test_udf_checkpoints_multiple_calls_same_job(
    test_session, monkeypatch, nums_dataset
):
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    # Track how many times the mapper is called
    call_count = {"count": 0}

    def add_ten(num) -> int:
        call_count["count"] += 1
        return num + 10

    chain = dc.read_dataset("nums", session=test_session).map(
        plus_ten=add_ten, output=int
    )

    reset_session_job_state()

    # First count() - should execute UDF
    assert chain.count() == 6
    first_calls = call_count["count"]
    assert first_calls == 6, "Mapper should be called 6 times on first count()"

    # Second count() - should reuse checkpoint within same job
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper should NOT be called on second count()"

    # Third count() - should still reuse checkpoint
    call_count["count"] = 0
    assert chain.count() == 6
    assert call_count["count"] == 0, "Mapper should NOT be called on third count()"

    # Other operations like to_list() should also reuse checkpoint
    call_count["count"] = 0
    result = chain.order_by("num").to_list("plus_ten")
    assert result == [(11,), (12,), (13,), (14,), (15,), (16,)]
    assert call_count["count"] == 0, "Mapper should NOT be called on to_list()"


def test_udf_shared_tables_naming(test_session, monkeypatch, nums_dataset):
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))

    # Record initial UDF tables (from nums_dataset fixture which uses read_values
    # internally)
    initial_udf_tables = set(warehouse.db.list_tables(prefix="udf_"))

    def get_udf_tables():
        tables = set(warehouse.db.list_tables(prefix="udf_"))
        return sorted(tables - initial_udf_tables)

    def square_num(num) -> int:
        return num * num

    chain = dc.read_dataset("nums", session=test_session).map(
        squared=square_num, output=int
    )

    # -------------- FIRST RUN -------------------
    reset_session_job_state()
    chain.count()
    first_job_id = test_session.get_or_create_job().id

    # Get checkpoints from first run to construct expected table names
    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 2

    # Checkpoints are ordered by creation, so first is hash_before, second is hash_after
    hash_before = checkpoints[0].hash
    hash_after = checkpoints[1].hash

    # Construct expected shared table names (no job_id in names)
    expected_udf_tables = sorted(
        [
            f"udf_{hash_before}_input",
            f"udf_{hash_after}_output",
        ]
    )

    assert get_udf_tables() == expected_udf_tables

    # -------------- SECOND RUN -------------------
    reset_session_job_state()
    chain.count()
    assert get_udf_tables() == expected_udf_tables


def test_udf_continue_from_partial(test_session, monkeypatch, nums_dataset):
    """Test continuing UDF execution from partial output table in unsafe mode.

    Uses settings(batch_size=2) to ensure multiple batches are committed, allowing
    partial results to persist even when UDF fails midway.
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    monkeypatch.setenv("DATACHAIN_UDF_CHECKPOINT_MODE", "unsafe")

    # Track which numbers have been processed and which run we're on
    processed_nums = []
    run_count = {"count": 0}

    def process_with_failure(num) -> int:
        """Process numbers but fail on num=4 in first run only."""
        processed_nums.append(num)
        if num == 4 and run_count["count"] == 0:
            raise Exception(f"Simulated failure on num={num}")
        return num * 10

    # -------------- FIRST RUN (FAILS AFTER FIRST BATCH) -------------------
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=2)
        .map(result=process_with_failure, output=int)
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("results")

    first_job_id = test_session.get_or_create_job().id

    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 1
    hash_before = checkpoints[0].hash

    # Verify partial output table exists
    partial_table_name = UDFStep.partial_output_table_name(first_job_id, hash_before)
    assert warehouse.db.has_table(partial_table_name)

    # Verify partial table has first batch (2 rows)
    partial_table = warehouse.get_table(partial_table_name)
    partial_count_query = sa.select(sa.func.count()).select_from(partial_table)
    assert warehouse.db.execute(partial_count_query).fetchone()[0] == 2

    # -------------- SECOND RUN (CONTINUE IN UNSAFE MODE) -------------------
    reset_session_job_state()

    # Clear processed list and increment run count to allow num=5 to succeed
    processed_nums.clear()
    run_count["count"] += 1

    # Now it should complete successfully
    chain.save("results")

    checkpoints = sorted(
        catalog.metastore.list_checkpoints(test_session.get_or_create_job().id),
        key=lambda c: c.created_at,
    )
    assert len(checkpoints) == 3
    assert warehouse.db.has_table(UDFStep.output_table_name(checkpoints[1].hash))

    # Verify all rows were processed
    assert (
        dc.read_dataset("results", session=test_session)
        .order_by("num")
        .to_list("result")
    ) == [(10,), (20,), (30,), (40,), (50,), (60,)]

    # Verify only unprocessed rows were processed in second run
    # First run with batch_size=2 commits: [1,2] (batch 1), then fails on row 4
    # So partial table has rows 1-2, second run processes rows 3,4,5,6
    assert processed_nums == [3, 4, 5, 6]
