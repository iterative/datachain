from collections.abc import Iterator

import pytest

import datachain as dc
from datachain.lib.file import File


def test_complex_signal_partition_by_file(test_session):
    """Test partitioning by File objects using column names."""

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

    # Test partitioning by File column name (automatically expands to unique keys)
    result = chain.agg(
        my_agg,
        params=("file", "amount"),
        output={"file": File, "total": int},
        partition_by="file",
    )

    # Verify the results
    files = result.to_values("file")
    totals = result.to_values("total")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(files) == 3
    assert len(totals) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = [f.path for f in files]
    assert "file1.txt" in file_paths
    assert "file2.txt" in file_paths
    assert "file3.txt" in file_paths

    # Check total amounts - create mapping from path to total
    path_to_total = {f.path: total for f, total in zip(files, totals)}
    assert path_to_total["file1.txt"] == 40  # 10 + 30 (grouped)
    assert path_to_total["file2.txt"] == 20
    assert path_to_total["file3.txt"] == 40


def test_complex_signal_partition_by_mixed(test_session):
    """Test partitioning by mixed types (complex signal column and string)."""

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

    def my_agg(
        files: list[File], categories: list[str], amounts: list[int]
    ) -> Iterator[tuple[File, str, int]]:
        yield files[0], categories[0], sum(amounts)

    # Test partitioning by both File column and string column
    result = chain.agg(
        my_agg,
        params=("file", "category", "amount"),
        output={"file": File, "category": str, "total": int},
        partition_by=["file", "category"],
    )

    files = result.to_values("file")
    categories = result.to_values("category")
    totals = result.to_values("total")

    # We should have 2 groups: (file1.txt, A), (file2.txt, B)
    assert len(files) == 2
    assert len(categories) == 2
    assert len(totals) == 2

    # Check grouping by both file and category
    groups = {(f.path, cat): total for f, cat, total in zip(files, categories, totals)}
    assert groups[("file1.txt", "A")] == 40  # 10 + 30
    assert groups[("file2.txt", "B")] == 20


def test_complex_signal_partition_by_error_handling(test_session):
    """Test error handling for invalid column names."""
    from datachain.lib.signal_schema import SignalResolvingError

    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )

    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield (sum(values),)

    # Test with non-existent column name
    with pytest.raises(SignalResolvingError, match="cannot resolve signal name 'nonexistent_column': is not found"):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by="nonexistent_column",
        ).to_records()


def test_complex_signal_partition_by_not_in_schema(test_session):
    """Test error handling when column name is not in schema."""
    from datachain.lib.signal_schema import SignalResolvingError

    chain = dc.read_values(
        value=[1, 2, 3],
        session=test_session,
    )

    def my_agg(values: list[int]) -> Iterator[tuple[int]]:
        yield (sum(values),)

    # Test with column name not in schema
    with pytest.raises(SignalResolvingError, match="cannot resolve signal name 'file': is not found"):
        chain.agg(
            my_agg,
            params=("value",),
            output={"total": int},
            partition_by="file",  # file column is not in this schema
        ).to_records()


def test_complex_signal_group_by_file(test_session):
    """Test group_by with File objects using column names."""

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

    # Test group_by with File column name (automatically expands to unique keys)
    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="file",
    )

    # Verify the results - after group_by, the schema contains flattened columns
    records = result.to_records()
    totals = result.to_values("total")
    counts = result.to_values("count")

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(records) == 3
    assert len(totals) == 3
    assert len(counts) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = [record['file__path'] for record in records]
    assert "file1.txt" in file_paths
    assert "file2.txt" in file_paths
    assert "file3.txt" in file_paths

    # Check total amounts
    path_to_total = {record['file__path']: record['total'] for record in records}
    assert path_to_total["file1.txt"] == 40  # 10 + 30 (grouped)
    assert path_to_total["file2.txt"] == 20
    assert path_to_total["file3.txt"] == 40

    # Check counts
    path_to_count = {record['file__path']: record['count'] for record in records}
    assert path_to_count["file1.txt"] == 2  # Two instances
    assert path_to_count["file2.txt"] == 1
    assert path_to_count["file3.txt"] == 1


def test_complex_signal_group_by_mixed(test_session):
    """Test group_by with mixed types (complex signal column and string)."""

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

    # Test group_by with both File column and string column
    result = chain.group_by(
        total=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by=["file", "category"],
    )

    records = result.to_records()
    categories = result.to_values("category")
    totals = result.to_values("total")

    # We should have 2 groups: (file1.txt, A), (file2.txt, B)
    assert len(records) == 2
    assert len(categories) == 2
    assert len(totals) == 2

    # Check grouping by both file and category
    groups = {(record['file__path'], record['category']): record['total'] for record in records}
    assert groups[("file1.txt", "A")] == 40  # 10 + 30
    assert groups[("file2.txt", "B")] == 20


def test_complex_signal_deep_nesting(test_session):
    """Test complex signals with deep nesting (3+ levels)."""
    from typing import ClassVar
    from pydantic import BaseModel
    
    # Create nested models with 3+ levels
    class NestedLevel1(BaseModel):
        name: str
        value: int
        
        _unique_id_keys: ClassVar[list[str]] = ['name', 'value']
    
    class NestedLevel2(BaseModel):
        category: str
        level1: NestedLevel1
        
        _unique_id_keys: ClassVar[list[str]] = ['category', 'level1']
    
    class NestedLevel3(BaseModel):
        id: str
        level2: NestedLevel2
        total: float
        
        _unique_id_keys: ClassVar[list[str]] = ['id', 'level2']
    
    # Create test data with deep nesting
    nested_data = [
        NestedLevel3(
            id="item1",
            level2=NestedLevel2(
                category="A",
                level1=NestedLevel1(name="test1", value=10)
            ),
            total=100.0
        ),
        NestedLevel3(
            id="item2", 
            level2=NestedLevel2(
                category="B",
                level1=NestedLevel1(name="test2", value=20)
            ),
            total=200.0
        ),
        NestedLevel3(
            id="item1",  # Same as first item
            level2=NestedLevel2(
                category="A",
                level1=NestedLevel1(name="test1", value=10)
            ),
            total=150.0
        ),
    ]
    
    amounts = [10, 20, 30]
    
    chain = dc.read_values(
        nested=nested_data,
        amount=amounts,
        session=test_session,
    )
    
    # Test group_by with deeply nested complex signal
    result = chain.group_by(
        total_amount=dc.func.sum("amount"),
        count=dc.func.count(),
        partition_by="nested",
    )
    
    records = result.to_records()
    total_amounts = result.to_values("total_amount")
    
    # Should have 2 groups (item1 appears twice with same structure)
    assert len(records) == 2
    assert len(total_amounts) == 2
    
    # Check grouping by nested structure
    groups = {record['nested__id']: record['total_amount'] for record in records}
    assert groups["item1"] == 40  # 10 + 30 (grouped by all nested fields)
    assert groups["item2"] == 20
