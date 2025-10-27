import pytest

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


@pytest.mark.parametrize(
    "batch_size,expected_partial_count,expected_unprocessed",
    [
        # Fail on row 4: batch 1 [1,2] commits, batch 2 fails on row 4
        (2, 2, [3, 4, 5, 6]),
        # Fail on row 4: batch 1 [1,2,3] not full, fails before commit
        (3, 0, [1, 2, 3, 4, 5, 6]),
        # Fail on row 4: batch 1 [1,2,3] not full, fails before commit
        (5, 0, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_udf_signals_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    expected_partial_count,
    expected_unprocessed,
):
    """Test continuing UDF execution from partial output table in unsafe mode.

    Tests with different batch sizes to ensure partial results are correctly handled
    regardless of batch boundaries.
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
        .settings(batch_size=batch_size)
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

    # Verify partial table has expected number of rows based on batch_size
    partial_table = warehouse.get_table(partial_table_name)
    assert warehouse.table_rows_count(partial_table) == expected_partial_count

    # -------------- SECOND RUN (CONTINUE IN UNSAFE MODE) -------------------
    reset_session_job_state()

    # Clear processed list and increment run count to allow num=4 to succeed
    processed_nums.clear()
    run_count["count"] += 1

    # Now it should complete successfully
    chain.save("results")

    second_job_id = test_session.get_or_create_job().id
    checkpoints = sorted(
        catalog.metastore.list_checkpoints(second_job_id),
        key=lambda c: c.created_at,
    )
    assert len(checkpoints) == 3
    output_table_name = UDFStep.output_table_name(checkpoints[1].hash)
    assert warehouse.db.has_table(output_table_name)

    # Verify all rows were processed
    assert (
        dc.read_dataset("results", session=test_session)
        .order_by("num")
        .to_list("result")
    ) == [(10,), (20,), (30,), (40,), (50,), (60,)]

    # Verify only unprocessed rows were processed in second run
    assert processed_nums == expected_unprocessed


@pytest.mark.parametrize(
    "batch_size,expected_partial_output_count,"
    "expected_processed_input_count,expected_unprocessed",
    [
        # batch_size=2: Small batches ensure multiple commits before failure
        #   Input 1 yields [10, 1] → batch 1 commits (2 outputs)
        #   Input 2 yields [20, 4] → batch 2 commits (2 outputs)
        #   Input 3 starts yielding but input 4 fails → batch incomplete
        (2, 4, 2, [3, 4, 5, 6]),
        # batch_size=10: Large batch means no commits before failure
        #   All 6 outputs from inputs 1,2,3 fit in incomplete first batch
        #   Input 4 fails before batch commits → 0 outputs, 0 inputs saved
        (10, 0, 0, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_udf_generator_continue_from_partial(
    test_session,
    monkeypatch,
    nums_dataset,
    batch_size,
    expected_partial_output_count,
    expected_processed_input_count,
    expected_unprocessed,
):
    """Test continuing RowGenerator from partial output in unsafe mode.

    RowGenerator differs from UDFSignal because:
    - One input can generate multiple outputs
    - Output rows have different sys__ids than input rows
    - Uses a separate processed table to track which inputs are processed

    Tests with different batch sizes to ensure processed table correctly
    tracks inputs only after ALL their outputs have been committed.
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    monkeypatch.setenv("DATACHAIN_UDF_CHECKPOINT_MODE", "unsafe")

    # Track which numbers have been processed and which run we're on
    processed_nums = []
    run_count = {"count": 0}

    class GeneratorWithFailure(dc.Generator):
        """Generator yielding 2 outputs per input, fails on num=4 in run 1."""

        def process(self, num):
            processed_nums.append(num)
            if num == 4 and run_count["count"] == 0:
                raise Exception(f"Simulated failure on num={num}")
            # Generate 2 outputs per input: the number and its square
            yield num * 10
            yield num * num

    # -------------- FIRST RUN (FAILS ON INPUT 4) -------------------
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=batch_size)
        .gen(value=GeneratorWithFailure(), output=int)
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("gen_results")

    first_job_id = test_session.get_or_create_job().id

    checkpoints = list(catalog.metastore.list_checkpoints(first_job_id))
    assert len(checkpoints) == 1
    hash_before = checkpoints[0].hash

    # Verify partial output table exists
    partial_table_name = UDFStep.partial_output_table_name(first_job_id, hash_before)
    assert warehouse.db.has_table(partial_table_name)

    # Verify partial table has expected number of outputs
    partial_table = warehouse.get_table(partial_table_name)
    assert warehouse.table_rows_count(partial_table) == expected_partial_output_count

    # Verify processed table exists and tracks fully processed inputs
    # An input is marked as processed only after ALL outputs committed
    processed_table_name = UDFStep.processed_table_name(first_job_id, hash_before)
    assert warehouse.db.has_table(processed_table_name)
    processed_table = warehouse.get_table(processed_table_name)
    assert warehouse.table_rows_count(processed_table) == expected_processed_input_count

    # -------------- SECOND RUN (CONTINUE IN UNSAFE MODE) -------------------
    reset_session_job_state()

    # Clear processed list and increment run count
    processed_nums.clear()
    run_count["count"] += 1

    # Now it should complete successfully
    chain.save("gen_results")

    second_job_id = test_session.get_or_create_job().id
    checkpoints = sorted(
        catalog.metastore.list_checkpoints(second_job_id),
        key=lambda c: c.created_at,
    )
    assert len(checkpoints) == 3
    output_table_name = UDFStep.output_table_name(checkpoints[1].hash)
    assert warehouse.db.has_table(output_table_name)

    # Verify all outputs were generated
    # 6 inputs x 2 outputs each = 12 total outputs
    result = (
        dc.read_dataset("gen_results", session=test_session)
        .order_by("value")
        .to_list("value")
    )
    expected = [
        (1,),
        (10,),  # num=1: 1 (1²), 10 (1x10)
        (4,),
        (20,),  # num=2: 4 (2²), 20 (2x10)
        (9,),
        (30,),  # num=3: 9 (3²), 30 (3x10)
        (16,),
        (40,),  # num=4: 16 (4²), 40 (4x10)
        (25,),
        (50,),  # num=5: 25 (5²), 50 (5x10)
        (36,),
        (60,),  # num=6: 36 (6²), 60 (6x10)
    ]
    assert sorted(result) == sorted(expected)

    # Verify only unprocessed inputs were processed in second run
    assert sorted(processed_nums) == sorted(expected_unprocessed)


@pytest.mark.xfail(
    reason="Known limitation: inputs that yield nothing are not tracked "
    "in processed table"
)
def test_generator_yielding_nothing(test_session, monkeypatch, nums_dataset):
    """Test that generator correctly handles inputs that yield zero outputs."""
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    monkeypatch.setenv("DATACHAIN_UDF_CHECKPOINT_MODE", "unsafe")

    processed = []

    class SelectiveGenerator(dc.Generator):
        """Generator that only yields outputs for even numbers."""

        def process(self, num):
            processed.append(num)
            if num == 3:
                raise Exception("Simulated failure")
            if num % 2 == 0:  # Only even numbers yield outputs
                yield num * 10

    # First run - fails on num=3
    reset_session_job_state()
    chain = dc.read_dataset("nums", session=test_session).gen(
        value=SelectiveGenerator(), output=int
    )

    with pytest.raises(Exception, match="Simulated failure"):
        chain.save("results")

    first_job_id = test_session.get_or_create_job().id
    first_checkpoints = list(
        test_session.catalog.metastore.list_checkpoints(first_job_id)
    )
    hash_before = first_checkpoints[0].hash

    # Verify processed table tracks inputs that yielded nothing
    warehouse = test_session.catalog.warehouse
    processed_table_name = UDFStep.processed_table_name(first_job_id, hash_before)
    assert warehouse.db.has_table(processed_table_name)
    processed_table = warehouse.get_table(processed_table_name)
    processed_count = warehouse.table_rows_count(processed_table)
    # Inputs 1,2 were processed (1 yielded nothing, 2 yielded one output)
    assert processed_count == 2

    # Second run - should skip already processed inputs
    reset_session_job_state()
    processed.clear()
    chain.save("results")

    # Only inputs 3,4,5,6 should be processed
    assert processed == [3, 4, 5, 6]
    # Result should only have even numbers x 10
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    assert result == [(20,), (40,), (60,)]


def test_multiple_udf_chain_continue(test_session, monkeypatch, nums_dataset):
    """Test continuing from partial with multiple UDFs in chain.

    When mapper fails, only mapper's partial table exists. On retry, mapper
    completes and gen runs from scratch.
    """
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", str(False))
    monkeypatch.setenv("DATACHAIN_UDF_CHECKPOINT_MODE", "unsafe")

    map_processed = []
    gen_processed = []

    def mapper(num: int) -> int:
        map_processed.append(num)
        # Fail on first encounter of num=4 (when counter is exactly 4)
        if num == 4 and len(map_processed) == 4:
            raise Exception("Map failure")
        return num * 2

    class Doubler(dc.Generator):
        def process(self, doubled):
            gen_processed.append(doubled)
            yield doubled
            yield doubled

    # First run - fails in mapper
    # batch_size=2: processes [1,2] (commits), then [3,4] (fails on 4)
    reset_session_job_state()
    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(batch_size=2)
        .map(doubled=mapper)
        .gen(value=Doubler(), output=int)
    )

    with pytest.raises(Exception, match="Map failure"):
        chain.save("results")

    # Second run - completes successfully
    # Mapper continues from partial [1,2], processes [3,4,5,6]
    # Then gen runs on all 6 outputs from mapper
    reset_session_job_state()
    chain.save("results")

    # Verify mapper was only called on unprocessed rows in second run
    # First run: [1,2,3,4], second run: [3,4,5,6] (continues from partial [1,2])
    # Total: [1,2,3,4,3,4,5,6]
    assert len(map_processed) == 8

    # Verify gen processed all mapper outputs
    assert len(gen_processed) == 6

    # Verify final result has all values doubled twice
    result = sorted(dc.read_dataset("results", session=test_session).to_list("value"))
    # Each of 6 inputs → doubled by map → doubled by gen = 12 outputs
    # Values: [2,4,6,8,10,12] each appearing twice
    expected = sorted([(i,) for i in [2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12]])
    assert result == expected
