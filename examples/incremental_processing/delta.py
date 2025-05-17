#!/usr/bin/env python
"""
File Generator Script using DataChain Delta

This script demonstrates:
1. Creating numbered text files in a 'test' directory
2. Using DataChain's delta flag for incremental dataset processing

Each execution:
- Creates a new numbered file in the 'test' directory
- Updates a DataChain dataset to track these files incrementally
"""

import re
import time

from utils import generate_next_file

import datachain as dc
from datachain import C, File


def extract_file_number(file: File) -> int:
    """Extract file number from the filename."""
    match = re.search(r"file-(\d+)\.txt", file.name)
    if match:
        return int(match.group(1))
    return -1


def process_files_with_delta():
    """
    Process files in the test directory using DataChain with delta mode.
    This demonstrates incremental processing - only new files are processed.
    """
    chain = (
        dc.read_storage("test/", update=True, delta=True, delta_on="file.path")
        .filter(C("file.path").glob("*.txt"))
        .map(file_number=extract_file_number)
        .map(content=lambda file: file.read_text())
        .map(processed_at=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
        .save(name="test_files")
    )

    # Show information about the dataset
    print(f"\nProcessed files. Total records: {chain.count()}")
    print("\nDataset versions:")
    test_dataset = dc.datasets().filter(C("name") == "test_files")

    for version in test_dataset.collect("version"):
        print(f"- Version: {version}")

    # Show the last 3 records to demonstrate the incremental processing
    print("\nLatest files processed:")
    chain.order_by("file_number", descending=True).limit(3).show()


if __name__ == "__main__":
    # Generate a new file
    new_file = generate_next_file()
    print(f"Created new file: {new_file}")

    # Process all new file with (delta update)
    process_files_with_delta()
