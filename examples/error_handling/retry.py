#!/usr/bin/env python
"""
Retry Processing Example with DataChain

This script demonstrates DataChain's retry functionality by:
1. Creating an initial in-memory dataset with sample data
2. Processing records with simulated errors
3. Using retry=True to reprocess failed records in a second pass

The example shows how DataChain can automatically retry processing
for records that previously failed, using match_on and retry_on parameters.
"""

from datetime import datetime, timezone

import datachain as dc
from datachain import C, DataModel


class ProcessingResult(DataModel):
    """Data model for processing results."""

    processed_content: str
    processed_at: str
    error: str


# Global counter to track processing attempts
_processing_attempts = {}


def process_data(data: dict) -> ProcessingResult:
    """
    Process a data record - initially fails for odd IDs, but succeeds on retry.
    In a real-world application, this could be data validation,
    transformation, or an ML model inference step.
    """
    item_id = data["id"]
    content = data["content"]

    # Track processing attempts for this item
    if item_id not in _processing_attempts:
        _processing_attempts[item_id] = 0
    _processing_attempts[item_id] += 1

    # Simulate an error for odd IDs on first attempt only
    # This will succeed on retry to demonstrate the retry functionality
    if item_id % 2 == 1 and _processing_attempts[item_id] == 1:
        return ProcessingResult(
            processed_content="",
            processed_at=datetime.now(tz=timezone.utc).isoformat(),
            error=f"Processing error for item {item_id} "
            f"(attempt {_processing_attempts[item_id]})",
        )

    # Successful processing
    return ProcessingResult(
        processed_content=content.upper(),
        processed_at=datetime.now(tz=timezone.utc).isoformat(),
        error="",
    )


def retry_processing_example():
    """
    Demonstrates retry processing using DataChain.
    """

    # Step 1: Create initial dataset with sample data
    print("Step 1: Creating initial dataset with sample data...")
    sample_ids = [1, 2, 3, 4, 5]
    sample_contents = [
        "first item",
        "second item",
        "third item",
        "fourth item",
        "fifth item",
    ]

    sample_data = [
        {"id": id_val, "content": content_val}
        for id_val, content_val in zip(sample_ids, sample_contents)
    ]

    initial_chain = dc.read_values(data=sample_data, in_memory=True).save(
        name="sample_data"
    )
    print(f"Created dataset with {initial_chain.count()} records\n")

    # Step 2: First processing pass - some records will fail
    print("Step 2: First processing pass (some records will fail)...")
    first_pass = (
        dc.read_dataset("sample_data")
        .map(result=process_data)
        .save(name="processed_data")
    )

    print(f"First pass completed. Total records: {first_pass.count()}")

    # Show results of first pass
    error_count = first_pass.filter(C("result.error") != "").count()
    success_count = first_pass.filter(C("result.error") == "").count()
    print(f"Successful: {success_count}, Failed: {error_count}")

    print("\nFirst pass results:")
    first_pass.show()

    # Step 3: Retry processing - only failed records will be reprocessed
    print("\nStep 3: Retry processing (failed records will be reprocessed)...")
    retry_chain = (
        dc.read_dataset(
            "sample_data",
            # Enable retry processing
            retry=True,
            # Match records based on the id field
            match_on="data.id",
            # Retry records where result.error field is not empty
            retry_on="result.error",
        )
        .map(result=process_data)
        .save(name="processed_data")
    )

    print(f"Retry pass completed. Total records: {retry_chain.count()}")

    # Show final results
    final_error_count = retry_chain.filter(C("result.error") != "").count()
    final_success_count = retry_chain.filter(C("result.error") == "").count()
    print(f"Final - Successful: {final_success_count}, Failed: {final_error_count}")

    print("\nFinal results after retry:")
    retry_chain.show()

    if final_error_count > 0:
        print(f"\nNote: {final_error_count} records still have errors after retry.")
        print(
            "In a real scenario, you could run this again or implement "
            "different retry logic."
        )
    else:
        print("\nAll records processed successfully after retry!")


if __name__ == "__main__":
    retry_processing_example()
