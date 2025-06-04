from datetime import datetime, timezone

import pytest

import datachain as dc
from datachain import C, DataModel
from datachain.error import DatasetVersionNotFoundError


class ProcessingResult(DataModel):
    """Data model for processing results in retry tests."""

    processed_content: str
    processed_at: str
    error: str


class TestRetryFunctionality:
    """Functional tests for the retry functionality in DataChain."""

    def test_retry_with_error_records(self, test_session):
        """Test retry functionality with records that have errors."""
        # Global counter to track processing attempts
        processing_attempts = {}

        def process_data_with_errors(id: int, content: str) -> ProcessingResult:
            """Process data that fails for odd IDs on first attempt."""
            item_id = id

            # Track processing attempts for this item
            if item_id not in processing_attempts:
                processing_attempts[item_id] = 0
            processing_attempts[item_id] += 1

            # Simulate an error for odd IDs on first attempt only
            if item_id % 2 == 1 and processing_attempts[item_id] == 1:
                return ProcessingResult(
                    processed_content="",
                    processed_at=datetime.now(tz=timezone.utc).isoformat(),
                    error=f"Processing error for item {item_id}",
                )

            # Successful processing
            return ProcessingResult(
                processed_content=content.upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",
            )

        # Create initial dataset with sample data
        sample_ids = [1, 2, 3, 4]
        sample_contents = ["first item", "second item", "third item", "fourth item"]

        dc.read_values(
            id=sample_ids, content=sample_contents, session=test_session
        ).save("sample_data")

        # First processing pass - some records will fail
        first_pass = (
            dc.read_dataset("sample_data", session=test_session)
            .map(result=process_data_with_errors)
            .save("processed_data")
        )

        # Check that some records failed
        error_count = first_pass.filter(C("result.error") != "").count()
        success_count = first_pass.filter(C("result.error") == "").count()

        assert error_count == 2  # Items with id 1 and 3 should fail
        assert success_count == 2  # Items with id 2 and 4 should succeed

        # Retry processing - only failed records should be reprocessed
        retry_chain = (
            dc.read_dataset(
                "sample_data",
                session=test_session,
                retry=True,
                match_on="id",
                retry_on="result.error",
            )
            .map(result=process_data_with_errors)
            .save("processed_data")
        )

        # Check final results - all should succeed now
        final_error_count = retry_chain.filter(C("result.error") != "").count()
        final_success_count = retry_chain.filter(C("result.error") == "").count()

        assert final_error_count == 0  # No errors after retry
        assert final_success_count == 4  # All records should succeed

        # Verify that retry only processed failed records
        assert processing_attempts[1] == 2  # Odd ID, processed twice
        assert processing_attempts[2] == 1  # Even ID, processed once
        assert processing_attempts[3] == 2  # Odd ID, processed twice
        assert processing_attempts[4] == 1  # Even ID, processed once

    def test_retry_with_missing_records(self, test_session):
        """Test retry functionality with missing records."""
        # Create source dataset
        source_ids = [1, 2, 3]
        source_contents = ["first", "second", "third"]

        dc.read_values(
            id=source_ids, content=source_contents, session=test_session
        ).save("source_data")

        # Create partial result dataset (missing id=3)
        def simple_process(id: int, content: str) -> ProcessingResult:
            return ProcessingResult(
                processed_content=content.upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",
            )

        # Process only first 2 records
        partial_result = (
            dc.read_dataset("source_data", session=test_session)
            .filter(C("id") <= 2)
            .map(result=simple_process)
            .save("partial_result")
        )

        assert partial_result.count() == 2

        # Use retry with retry_missing=True to process missing records
        retry_chain = (
            dc.read_dataset(
                "source_data",
                session=test_session,
                retry=True,
                match_on="id",
                retry_on="result.error",
                retry_missing=True,
            )
            .map(result=simple_process)
            .save("partial_result")
        )

        # Should now have all 3 records
        assert retry_chain.count() == 3

        # Verify all records are present
        ids = set(retry_chain.collect("id"))
        assert ids == {1, 2, 3}

    def test_retry_no_records_to_retry(self, test_session):
        """Test retry when no records need to be retried."""
        # Create dataset with all successful records
        source_ids = [1, 2]
        source_contents = ["first", "second"]

        dc.read_values(
            id=source_ids, content=source_contents, session=test_session
        ).save("source_data")

        def successful_process(id: int, content: str) -> ProcessingResult:
            return ProcessingResult(
                processed_content=content.upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",  # No errors
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
                retry=True,
                match_on="id",
                retry_on="result.error",
            )
            .map(result=successful_process)
            .save("successful_data")
        )

        # Should not create version 1.0.1 since no retry was needed
        with pytest.raises(DatasetVersionNotFoundError):
            dc.read_dataset("successful_data", version="1.0.1", session=test_session)

    def test_retry_first_dataset_creation(self, test_session):
        """Test retry when dataset doesn't exist yet (first creation)."""
        source_ids = [1, 2]
        source_contents = ["first", "second"]

        dc.read_values(
            id=source_ids, content=source_contents, session=test_session
        ).save("source_data")

        def simple_process(id: int, content: str) -> ProcessingResult:
            return ProcessingResult(
                processed_content=content.upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",
            )

        # First run with retry enabled on non-existent dataset
        # Should process all records
        retry_chain = (
            dc.read_dataset(
                "source_data",
                session=test_session,
                retry=True,
                match_on="id",
                retry_on="result.error",
            )
            .map(result=simple_process)
            .save("new_dataset")
        )

        assert retry_chain.count() == 2
        assert retry_chain.filter(C("result.error") != "").count() == 0

    def test_retry_with_multiple_match_fields(self, test_session):
        """Test retry functionality with multiple fields for matching."""
        processing_attempts = {}

        def process_with_compound_key(
            category: str, id: int, content: str
        ) -> ProcessingResult:
            """Process data that fails based on compound key."""
            key = (category, id)

            if key not in processing_attempts:
                processing_attempts[key] = 0
            processing_attempts[key] += 1

            # Fail items where category="A" and id is odd on first attempt
            if category == "A" and id % 2 == 1 and processing_attempts[key] == 1:
                return ProcessingResult(
                    processed_content="",
                    processed_at=datetime.now(tz=timezone.utc).isoformat(),
                    error=f"Error for {category}-{id}",
                )

            return ProcessingResult(
                processed_content=f"{category}-{content}".upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",
            )

        # Create dataset with compound keys
        categories = ["A", "A", "B", "B"]
        ids = [1, 2, 1, 2]
        contents = ["first", "second", "third", "fourth"]

        dc.read_values(
            category=categories, id=ids, content=contents, session=test_session
        ).save("compound_source")

        # First pass
        first_pass = (
            dc.read_dataset("compound_source", session=test_session)
            .map(result=process_with_compound_key)
            .save("compound_result")
        )

        # Only A-1 should fail (category="A" and odd id)
        error_count = first_pass.filter(C("result.error") != "").count()
        assert error_count == 1

        # Retry with multiple match fields
        retry_chain = (
            dc.read_dataset(
                "compound_source",
                session=test_session,
                retry=True,
                match_on=["category", "id"],
                retry_on="result.error",
            )
            .map(result=process_with_compound_key)
            .save("compound_result")
        )

        # All should succeed after retry
        final_error_count = retry_chain.filter(C("result.error") != "").count()
        assert final_error_count == 0
        assert retry_chain.count() == 4

        # Verify retry logic
        assert processing_attempts[("A", 1)] == 2  # Failed then retried
        assert processing_attempts[("A", 2)] == 1  # Succeeded first time
        assert processing_attempts[("B", 1)] == 1  # Succeeded first time
        assert processing_attempts[("B", 2)] == 1  # Succeeded first time

    def test_retry_with_delta_functionality(self, test_session):
        """Test that retry and delta can work together."""
        # Create initial dataset
        source_ids = [1, 2]
        source_contents = ["first", "second"]

        dc.read_values(
            id=source_ids, content=source_contents, session=test_session
        ).save("delta_source_v1")

        def simple_process(id: int, content: str) -> ProcessingResult:
            return ProcessingResult(
                processed_content=content.upper(),
                processed_at=datetime.now(tz=timezone.utc).isoformat(),
                error="",
            )

        # Process with retry and delta enabled
        result = (
            dc.read_dataset(
                "delta_source_v1",
                session=test_session,
                retry=True,
                match_on="id",
                retry_on="result.error",
                delta=True,
                delta_on="id",
            )
            .map(result=simple_process)
            .save("delta_retry_result")
        )

        assert result.count() == 2

        # Add more data
        extended_ids = [1, 2, 3]
        extended_contents = ["first", "second", "third"]  # New record

        dc.read_values(
            id=extended_ids, content=extended_contents, session=test_session
        ).save("delta_source_v2")

        # Process again - should only process new record due to delta
        result_v2 = (
            dc.read_dataset(
                "delta_source_v2",
                session=test_session,
                retry=True,
                match_on="id",
                retry_on="result.error",
                delta=True,
                delta_on="id",
            )
            .map(result=simple_process)
            .save("delta_retry_result")
        )

        # Should have all 3 records now
        assert result_v2.count() == 3
