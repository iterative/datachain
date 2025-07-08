from typing import Iterator

import pytest

import datachain as dc
from datachain.lib.file import File


def test_complex_signal_partition_by_file(test_session):
    """Test partitioning by File objects directly."""
    
    # Create some test files
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="file3.txt", size=300),
    ]
    
    amounts = [10, 20, 30, 40]
    
    # Create a chain with File objects
    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=test_session,
    )
    
    def my_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[File, int]]:
        # Group files and sum amounts
        yield files[0], sum(amounts)
    
    # Test partitioning by File type directly
    result = chain.agg(
        my_agg,
        params=("file", "amount"),
        output={"file": File, "total": int},
        partition_by=File,
    )
    
    # Verify the results
    records = result.to_records()
    
    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(records) == 3
    
    # Check that files with same unique attributes are grouped together
    file_paths = [r["file"].path for r in records]
    assert "file1.txt" in file_paths
    assert "file2.txt" in file_paths
    assert "file3.txt" in file_paths
    
    # Check total amounts
    totals = {r["file"].path: r["total"] for r in records}
    assert totals["file1.txt"] == 40  # 10 + 30 (grouped)
    assert totals["file2.txt"] == 20
    assert totals["file3.txt"] == 40


def test_complex_signal_partition_by_mixed(test_session):
    """Test partitioning by mixed types (string and File)."""
    
    # Create test data
    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
    ]
    
    categories = ["A", "B", "A"]
    amounts = [10, 20, 30]
    
    chain = dc.read_values(
        file=files,
        category=categories,
        amount=amounts,
        session=test_session,
    )
    
    def my_agg(files: list[File], categories: list[str], amounts: list[int]) -> Iterator[tuple[File, str, int]]:
        yield files[0], categories[0], sum(amounts)
    
    # Test partitioning by both File and string column
    result = chain.agg(
        my_agg,
        params=("file", "category", "amount"),
        output={"file": File, "category": str, "total": int},
        partition_by=[File, "category"],
    )
    
    records = result.to_records()
    
    # We should have 3 groups: (file1.txt, A), (file2.txt, B), (file1.txt, A) -> 2 groups
    assert len(records) == 2
    
    # Check grouping by both file and category
    groups = {(r["file"].path, r["category"]): r["total"] for r in records}
    assert groups[("file1.txt", "A")] == 40  # 10 + 30
    assert groups[("file2.txt", "B")] == 20


def test_complex_signal_partition_by_error_handling(test_session):
    """Test error handling for invalid complex signal types."""
    
    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )
    
    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield sum(values),
    
    # Test with non-DataModel type
    with pytest.raises(ValueError, match="Complex signal type .* must be a DataModel subclass"):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by=int,  # int is not a DataModel
        ).to_records()


def test_complex_signal_partition_by_not_in_schema(test_session):
    """Test error handling when complex signal type is not in schema."""
    
    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )
    
    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield sum(values),
    
    # Test with DataModel type not in schema
    with pytest.raises(ValueError, match="Signal type .* not found in the current schema"):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by=File,  # File is not in this schema
        ).to_records()