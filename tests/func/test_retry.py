from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

import datachain as dc
from datachain import C, DataModel
from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from datachain import DataChain


class ProcessingResult(DataModel):
    """Data model for processing results in retry tests."""

    processed_content: str
    processed_at: str
    error: str
    attempt: int


def _process_with_errors(id: int, content: str, attempt: int) -> ProcessingResult:
    error = id % 2 == 1 and attempt == 1
    return ProcessingResult(
        processed_content="" if error else content.upper(),
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
        error=f"Processing error for item {id}" if error else "",
        attempt=attempt,
    )


def test_retry_with_error_records(test_session):
    """Test retry functionality with records that have errors."""

    def _run_processing(attempt: int) -> "DataChain":
        return (
            dc.read_dataset(
                "sample_data",
                delta=True,
                delta_on="id",
                delta_retry="result.error",
                session=test_session,
            )
            .setup(attempt=lambda: attempt)
            .map(result=_process_with_errors)
            .save("processed_data")
        )

    # First processing pass - some records will fail
    sample_ids = [1, 2, 3, 4]
    sample_contents = ["first item", "second item", "third item", "fourth item"]

    dc.read_values(id=sample_ids, content=sample_contents, session=test_session).save(
        "sample_data"
    )

    first_pass = _run_processing(1)

    # Check that some records failed
    error_count = first_pass.filter(C("result.error") != "").count()
    success_count = first_pass.filter(C("result.error") == "").count()

    assert error_count == 2
    assert success_count == 2

    # Retry processing - only failed records should be reprocessed
    _run_processing(2)

    final_result = dc.read_dataset("processed_data", session=test_session)
    results = final_result.to_list("id", "result.attempt", "result.error")
    assert set(results) == {(2, 1, ""), (4, 1, ""), (1, 2, ""), (3, 2, "")}


def test_retry_with_missing_records(test_session):
    """Test retry functionality with missing records."""
    # Create source dataset
    source_ids = [1, 2, 3]
    source_contents = ["first", "second", "third"]

    dc.read_values(id=source_ids, content=source_contents, session=test_session).save(
        "source_data"
    )

    def simple_process(id: int, content: str, attempt: int) -> ProcessingResult:
        return ProcessingResult(
            processed_content=content.upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error="",
            attempt=attempt,
        )

    # Process only first 2 records
    # Create partial result dataset (missing id=3)
    partial_result = (
        dc.read_dataset("source_data", session=test_session)
        .setup(attempt=lambda: 1)
        .filter(C("id") < 3)
        .map(result=simple_process)
        .save("partial_result")
    )

    assert partial_result.count() == 2

    # Use retry with delta_retry=True to process missing records
    retry_chain = (
        dc.read_dataset(
            "source_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry=True,
        )
        .setup(attempt=lambda: 2)
        .map(result=simple_process)
        .save("partial_result")
    )

    # Should now have all 3 records
    assert retry_chain.count() == 3

    # Verify all records are present
    ids = set(retry_chain.to_values("id"))
    assert ids == {1, 2, 3}

    final_first_attempts_count = retry_chain.filter(C("result.attempt") == 1).count()
    final_missing_attempts_count = retry_chain.filter(C("result.attempt") == 2).count()

    # Only missing records should have attempt 2
    assert final_missing_attempts_count == 1
    assert final_first_attempts_count == 2


def test_retry_no_records_to_retry(test_session):
    """Test retry when no records need to be retried."""
    # Create dataset with all successful records
    source_ids = [1, 2]
    source_contents = ["first", "second"]

    dc.read_values(id=source_ids, content=source_contents, session=test_session).save(
        "source_data"
    )

    def successful_process(id: int, content: str) -> ProcessingResult:
        return ProcessingResult(
            processed_content=content.upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error="",  # No errors
            attempt=1,
        )

    # First pass - all succeed
    first_pass = (
        dc.read_dataset("source_data", session=test_session)
        .map(result=successful_process)
        .save("successful_data")
    )

    assert first_pass.count() == 2
    assert first_pass.filter(C("result.error") != "").count() == 0

    # Retry - should not create a new version since no records need retry
    (
        dc.read_dataset(
            "source_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="result.error",
        )
        .map(result=successful_process)
        .save("successful_data")
    )

    # Should not create version 1.0.1 since no retry was needed
    with pytest.raises(DatasetNotFoundError):
        dc.read_dataset("successful_data", version="1.0.1", session=test_session)


def test_retry_first_dataset_creation(test_session):
    """Test retry when dataset doesn't exist yet (first creation)."""
    source_ids = [1, 2]
    source_contents = ["first", "second"]

    dc.read_values(id=source_ids, content=source_contents, session=test_session).save(
        "source_data"
    )

    def simple_process(id: int, content: str) -> ProcessingResult:
        return ProcessingResult(
            processed_content=content.upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error="",
            attempt=1,
        )

    # First run with retry enabled on non-existent dataset
    # Should process all records
    retry_chain = (
        dc.read_dataset(
            "source_data",
            session=test_session,
            delta=True,
            delta_on="id",
            delta_retry="result.error",
        )
        .map(result=simple_process)
        .save("new_dataset")
    )

    assert retry_chain.count() == 2
    assert retry_chain.filter(C("result.error") != "").count() == 0


def test_retry_with_multiple_match_fields(test_session):
    """Test retry functionality with multiple fields for matching."""

    def process_with_compound_key(
        category: str, id: int, content: str, attempt: int
    ) -> ProcessingResult:
        error = category == "A" and id % 2 == 1 and attempt == 1
        return ProcessingResult(
            processed_content="" if error else f"{category}-{content}".upper(),
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error=f"Error for {category}-{id}" if error else "",
            attempt=attempt,
        )

    # Create dataset with compound keys
    categories = ["A", "A", "B", "B"]
    ids = [1, 2, 1, 2]
    contents = ["first", "second", "third", "fourth"]

    dc.read_values(
        category=categories, id=ids, content=contents, session=test_session
    ).save("compound_source")

    first_pass = (
        dc.read_dataset("compound_source", session=test_session)
        .setup(attempt=lambda: 1)
        .map(result=process_with_compound_key)
        .save("compound_result")
    )

    # Only A-1 should fail (category="A" and odd id)
    results = list(first_pass.filter(C("result.error") != "").to_list("category", "id"))
    assert results == [("A", 1)]

    # Retry with multiple match fields
    (
        dc.read_dataset(
            "compound_source",
            session=test_session,
            delta=True,
            delta_on=["category", "id"],
            delta_retry="result.error",
        )
        .setup(attempt=lambda: 2)
        .map(result=process_with_compound_key)
        .save("compound_result")
    )

    final_result = dc.read_dataset("compound_result", session=test_session)
    results = final_result.to_list("category", "id", "result.attempt", "result.error")
    assert set(results) == {
        ("A", 2, 1, ""),
        ("B", 1, 1, ""),
        ("B", 2, 1, ""),
        ("A", 1, 2, ""),
    }


def test_retry_with_delta_functionality(test_session):
    """Test that retry and delta can work together."""

    def _run_processing(attempt: int) -> "DataChain":
        return (
            dc.read_dataset(
                "delta_source_v1",
                delta=True,
                delta_on="id",
                delta_retry="result.error",
                session=test_session,
            )
            .setup(attempt=lambda: attempt)
            .map(result=_process_with_errors)
            .save("delta_retry_result")
        )

    dc.read_values(id=[1, 2], content=["first", "second"], session=test_session).save(
        "delta_source_v1"
    )

    result = _run_processing(1)
    assert set(result.to_iter("id", "result.error")) == {
        (1, "Processing error for item 1"),
        (2, ""),
    }

    # Add more data
    # Process again - should only process new records and retry errors
    extended_ids = [1, 2, 3]
    extended_contents = ["first", "second", "third"]
    dc.read_values(
        id=extended_ids, content=extended_contents, session=test_session
    ).save("delta_source_v1")

    result_v2 = _run_processing(2)
    assert set(result_v2.to_iter("id", "result.error", "result.attempt")) == {
        (1, "", 2),
        (2, "", 1),
        (3, "", 2),
    }
