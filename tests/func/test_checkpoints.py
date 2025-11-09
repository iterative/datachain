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
    from datachain.query.dataset import UDFStep

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
