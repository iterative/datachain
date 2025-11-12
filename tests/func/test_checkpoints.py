from collections.abc import Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy.sql.schema import Table

import datachain as dc
from datachain.error import DatasetNotFoundError
from datachain.query.dataset import UDFStep
from tests.utils import reset_session_job_state


@pytest.fixture(autouse=True)
def mock_is_script_run(monkeypatch):
    """Mock is_script_run to return True for stable job names in tests."""
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


def get_partial_tables(test_session, generator=True) -> tuple[Table, Table]:
    """Helper function that returns partial udf tables left when UDF fails.

    Returns input_table and partial_output_table.
    Note: processed_table is no longer created - sys__input_id in partial_output_table
    tracks which inputs have been processed.
    """
    catalog = test_session.catalog
    warehouse = catalog.warehouse
    job_id = test_session.get_or_create_job().id
    checkpoints = list(catalog.metastore.list_checkpoints(job_id))
    assert len(checkpoints) == 1
    hash_input = checkpoints[0].hash

    # input table name
    input_table_name = UDFStep.input_table_name(job_id, hash_input)
    assert warehouse.db.has_table(input_table_name)
    input_table = warehouse.get_table(input_table_name)

    # partial output table name
    partial_table_name = UDFStep.partial_output_table_name(job_id, hash_input)
    assert warehouse.db.has_table(partial_table_name)
    partial_output_table = warehouse.get_table(partial_table_name)

    return input_table, partial_output_table


@pytest.mark.skipif(
    "os.environ.get('DATACHAIN_DISTRIBUTED')",
    reason="Checkpoints test skipped in distributed mode",
)
def test_checkpoints_parallel(test_session_tmpfile, monkeypatch):
    def mapper_fail(num) -> int:
        raise Exception("Error")

    test_session = test_session_tmpfile
    catalog = test_session.catalog

    dc.read_values(num=list(range(1000)), session=test_session).save("nums")

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


def test_udf_generator_continue_parallel(test_session_tmpfile, monkeypatch):
    """Test continuing RowGenerator from partial with parallel=True.

    This tests that processed table is properly passed through parallel
    execution path so that checkpoint recovery works correctly.
    """
    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse

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
    partial_output_table = warehouse.get_table(partial_table_name)

    # Verify sys__input_id has tracked some inputs
    processed_count_first = len(
        list(
            warehouse.db.execute(
                sa.select(sa.distinct(partial_output_table.c.sys__input_id))
            )
        )
    )
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


@pytest.mark.parametrize("parallel", [None, 2, 4, 6, 20])
def test_processed_table(test_session_tmpfile, parallel):
    """Test that processed table correctly tracks sys__ids with different parallel
    settings.

    This is a simple test that runs a UDF that fails partway through and verifies
    that the processed table contains exactly the sys__ids that were successfully
    processed (no duplicates, no missing values).

    Works with any warehouse (SQLite, ClickHouse, PostgreSQL) without assuming
    sequential sys__id values.
    """
    test_session = test_session_tmpfile
    catalog = test_session.catalog
    warehouse = catalog.warehouse

    def gen_numbers(num) -> Iterator[int]:
        """Generator function that fails on a specific input."""
        # Fail on input 7
        if num == 7:
            raise Exception(f"Simulated failure on num={num}")
        # Yield the number multiplied by 10
        yield num * 10

    # Create dataset with 100 numbers
    dc.read_values(num=list(range(1, 100)), session=test_session).save("nums")

    reset_session_job_state()

    # Build chain with optional parallel setting
    # ORDER BY ensures num=7 is encountered predictably (after processing 1-6)
    chain = (
        dc.read_dataset("nums", session=test_session)
        .order_by("num")
        .settings(batch_size=2)
    )
    if parallel is not None:
        chain = chain.settings(parallel=parallel)

    # Run UDF - should fail on num=7
    with pytest.raises(Exception):  # noqa: B017
        chain.gen(result=gen_numbers, output=int).save("results")

    _, partial_output_table = get_partial_tables(test_session)

    # Get distinct sys__input_id from partial output table to see which inputs were
    # processed
    query = sa.select(sa.distinct(partial_output_table.c.sys__input_id))
    processed_sys_ids = [row[0] for row in warehouse.db.execute(query)]

    # Verify no duplicates - this is the critical check for race conditions
    assert len(processed_sys_ids) == len(set(processed_sys_ids))
    # Verify we processed some but not all inputs (should have failed before completing)
    assert 0 < len(processed_sys_ids) < 100


@pytest.mark.parametrize("parallel", [2, 4, 6, 20])
def test_processed_table_data_integrity(test_session_tmpfile, parallel):
    """Test that processed table, input table, and output table are consistent after
    failure.

    Verifies that for a generator that yields n^2 for each input n:
    - Every sys__id in processed table has corresponding input in input table
    - Every processed input has correct output (n^2) in partial output table
    - No duplicate sys__ids in processed table (race condition check)
    - No missing or incorrect outputs
    """
    test_session = test_session_tmpfile
    warehouse = test_session.catalog.warehouse

    def gen_square(num) -> Iterator[int]:
        """Generator that yields n^2 for each input n."""
        # Fail on input 7
        if num == 50:
            raise Exception(f"Simulated failure on num={num}")
        yield num * num

    dc.read_values(num=list(range(1, 101)), session=test_session).save("nums")
    reset_session_job_state()

    chain = (
        dc.read_dataset("nums", session=test_session)
        .settings(parallel=parallel, batch_size=2)
        .gen(result=gen_square, output=int)
    )

    # Run UDF - should fail on num=7
    with pytest.raises(RuntimeError):
        chain.save("results")

    input_table, partial_output_table = get_partial_tables(test_session)

    # Get distinct sys__input_id from partial output table to see which inputs were
    # processed
    processed_sys_ids = [
        row[0]
        for row in warehouse.db.execute(
            sa.select(sa.distinct(partial_output_table.c.sys__input_id))
        )
    ]
    # Build mapping: sys__id -> input_value from input table
    input_data = {
        row[0]: row[1]
        for row in warehouse.db.execute(
            sa.select(input_table.c.sys__id, input_table.c.num)
        )
    }
    # output values in partial output table
    outputs = [
        row[0] for row in warehouse.db.execute(sa.select(partial_output_table.c.result))
    ]

    # Verify no duplicates
    assert len(processed_sys_ids) == len(set(processed_sys_ids))

    # Verify no duplicates
    assert len(set(outputs)) == len(outputs)

    # check that there are same number of records in processed table as in output
    assert len(processed_sys_ids) == len(outputs)

    # Verify each processed sys__id has correct input and output
    for sys_id in processed_sys_ids:
        # Check input exists for this sys__id
        assert sys_id in input_data

        # Verify output value is correct (n^2)
        input_val = input_data[sys_id]
        expected_output = input_val * input_val

        assert expected_output in outputs, (
            f"For sys__id {sys_id}: input={input_val}, "
            f"expected output={expected_output}, "
            f"not found in partial output"
        )

    # Verify we processed some inputs (don't check exact count - varies by warehouse)
    assert len(processed_sys_ids) > 0, "Expected some processing before failure"
