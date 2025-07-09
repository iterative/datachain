from collections.abc import Iterator
from pathlib import PurePosixPath

import pytest

import datachain as dc
from datachain import DataModel
from datachain.lib.file import File
from tests.utils import sorted_dicts


@pytest.mark.parametrize("parallel", [1, 2])
def test_agg(catalog_tmpfile, parallel):
    from datachain import func

    session = catalog_tmpfile.session

    def process(files: list[str]) -> Iterator[tuple[str, int]]:
        yield str(PurePosixPath(files[0]).parent), len(files)

    ds = (
        dc.read_values(
            filename=(
                "cats/cat1",
                "cats/cat2",
                "dogs/dog1",
                "dogs/dog2",
                "dogs/dog3",
                "dogs/others/dog4",
            ),
            session=session,
        )
        .settings(parallel=parallel)
        .agg(
            process,
            params=["filename"],
            output={"parent": str, "count": int},
            partition_by=func.path.parent("filename"),
        )
        .save("my-ds")
    )

    assert sorted_dicts(ds.to_records(), "parent") == sorted_dicts(
        [
            {"parent": "cats", "count": 2},
            {"parent": "dogs", "count": 3},
            {"parent": "dogs/others", "count": 1},
        ],
        "parent",
    )


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize(
    "offset,limit,files",
    [
        (None, 1000, [f"file{i:02d}" for i in range(100)]),
        (None, 3, ["file00", "file01", "file02"]),
        (0, 3, ["file00", "file01", "file02"]),
        (97, 1000, ["file97", "file98", "file99"]),
        (1, 2, ["file01", "file02"]),
        (50, 3, ["file50", "file51", "file52"]),
        (None, 0, []),
        (50, 0, []),
    ],
)
def test_agg_offset_limit(catalog_tmpfile, parallel, offset, limit, files):
    def process(filename: list[str]) -> Iterator[tuple[str, int]]:
        yield filename[0], len(filename)

    ds = dc.read_values(
        filename=[f"file{i:02d}" for i in range(100)],
        value=list(range(100)),
        session=catalog_tmpfile.session,
    )
    if offset is not None:
        ds = ds.offset(offset)
    if limit is not None:
        ds = ds.limit(limit)
    ds = (
        ds.settings(parallel=parallel)
        .agg(
            process,
            output={"filename": str, "count": int},
            partition_by="filename",
        )
        .save("my-ds")
    )

    records = list(ds.to_records())
    assert len(records) == len(files)
    assert all(row["count"] == 1 for row in records)
    assert sorted(row["filename"] for row in records) == sorted(files)


@pytest.mark.parametrize("parallel", [1, 2])
@pytest.mark.parametrize("sample", [0, 1, 3, 10, 50, 100])
def test_agg_sample(catalog_tmpfile, parallel, sample):
    def process(filename: list[str]) -> Iterator[tuple[str, int]]:
        yield filename[0], len(filename)

    ds = (
        dc.read_values(
            filename=[f"file{i:02d}" for i in range(100)],
            session=catalog_tmpfile.session,
        )
        .sample(sample)
        .settings(parallel=parallel)
        .agg(
            process,
            output={"filename": str, "count": int},
            partition_by="filename",
        )
        .save("my-ds")
    )

    records = list(ds.to_records())
    assert len(records) == sample
    assert all(row["count"] == 1 for row in records)


@pytest.mark.parametrize("parallel", [1, 2])
def test_partition_by_file(catalog_tmpfile, parallel):
    """Test partitioning by File objects directly."""

    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="file3.txt", size=300),
    ]
    amounts = [10, 20, 30, 40]

    chain = dc.read_values(
        file=files,
        amount=amounts,
        session=catalog_tmpfile.session,
    ).settings(parallel=parallel)

    def test_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[File, int]]:
        yield files[0], sum(amounts)

    # Test partitioning by File type directly
    result = chain.agg(
        test_agg,
        params=("file", "amount"),
        output={"file": File, "total": int},
        partition_by="file",
    )
    records = result.to_records()

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(records) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = {r["file__path"] for r in records}
    assert file_paths == {"file1.txt", "file2.txt", "file3.txt"}

    # Check total amounts
    totals = {r["file__path"]: r["total"] for r in records}
    assert totals == {
        "file1.txt": 40,  # 10 + 30 (grouped)
        "file2.txt": 20,
        "file3.txt": 40,
    }


@pytest.mark.parametrize("parallel", [1, 2])
def test_partition_by_file_and_string(catalog_tmpfile, parallel):
    """Test partitioning by mixed types (File and string)."""

    files = [
        File(source="s3://bucket", path="file1.txt", size=100),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
        File(source="s3://bucket", path="file2.txt", size=200),
        File(source="s3://bucket", path="file1.txt", size=100),  # duplicate
    ]
    categories = ["A", "B", "B", "A"]
    amounts = [10, 20, 30, 40]

    chain = dc.read_values(
        file=files,
        category=categories,
        amount=amounts,
        session=catalog_tmpfile.session,
    ).settings(parallel=parallel)

    def test_agg(
        files: list[File], categories: list[str], amounts: list[int]
    ) -> Iterator[tuple[File, str, int]]:
        yield files[0], categories[0], sum(amounts)

    # Test partitioning by both File and string column
    result = chain.agg(
        test_agg,
        params=("file", "category", "amount"),
        output={"file": File, "category": str, "total": int},
        partition_by=("file", "category"),
    )
    records = result.to_records()

    # We should have 3 groups:
    # (file1.txt, A), (file1.txt, B), (file2.txt, B) -> 3 groups
    assert len(records) == 3

    # Check grouping by both file and category
    groups = {(r["file__path"], r["category"]): r["total"] for r in records}
    assert groups == {
        ("file1.txt", "A"): 50,  # 10 + 40
        ("file1.txt", "B"): 20,  # 20
        ("file2.txt", "B"): 30,  # 30
    }


@pytest.mark.parametrize("parallel", [1, 2])
def test_partition_by_nested_file(catalog_tmpfile, parallel):
    """Test partitioning by File objects directly."""

    class Signal(DataModel):
        file: File
        amount: int

    signals = [
        Signal(file=File(source="s3://bucket", path="file1.txt", size=100), amount=10),
        Signal(file=File(source="s3://bucket", path="file2.txt", size=200), amount=20),
        Signal(
            file=File(source="s3://bucket", path="file1.txt", size=100), amount=30
        ),  # duplicate
        Signal(file=File(source="s3://bucket", path="file3.txt", size=300), amount=40),
    ]
    chain = dc.read_values(
        signal=signals,
        session=catalog_tmpfile.session,
    ).settings(parallel=parallel)

    def test_agg(files: list[File], amounts: list[int]) -> Iterator[tuple[File, int]]:
        yield files[0], sum(amounts)

    # Test partitioning by File type directly
    result = chain.agg(
        test_agg,
        params=("signal.file", "signal.amount"),
        output={"file": File, "total": int},
        partition_by="signal.file",
    )
    records = result.to_records()

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(records) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = {r["file__path"] for r in records}
    assert file_paths == {"file1.txt", "file2.txt", "file3.txt"}

    # Check total amounts
    totals = {r["file__path"]: r["total"] for r in records}
    assert totals == {
        "file1.txt": 40,  # 10 + 30 (grouped)
        "file2.txt": 20,
        "file3.txt": 40,
    }


@pytest.mark.parametrize("parallel", [1, 2])
def test_partition_by_inherited_file(catalog_tmpfile, parallel):
    """
    Test partitioning by inherited File objects.
    Additional fields in the inherited class should be used for grouping.
    """

    class MyFile(File):
        amount: int

    my_files = [
        MyFile(source="s3://bucket", path="file1.txt", size=100, amount=10),
        MyFile(
            source="s3://bucket", path="file1.txt", size=100, amount=20
        ),  # not a duplicate
        MyFile(
            source="s3://bucket", path="file1.txt", size=100, amount=10
        ),  # duplicate
        MyFile(source="s3://bucket", path="file3.txt", size=300, amount=40),
    ]
    chain = dc.read_values(
        file=my_files,
        session=catalog_tmpfile.session,
    ).settings(parallel=parallel)

    def test_agg(files: list[MyFile]) -> Iterator[tuple[MyFile, int, int]]:
        yield files[0], sum(f.amount for f in files), len(files)

    # Test partitioning by File type directly
    result = chain.agg(
        test_agg,
        params=("file",),
        output={"file": MyFile, "total": int, "cnt": int},
        partition_by="file",
    )
    records = result.to_records()

    # We should have 3 groups (file1.txt appears twice, so should be grouped)
    assert len(records) == 3

    # Check that files with same unique attributes are grouped together
    file_paths = [r["file__path"] for r in records]
    assert sorted(file_paths) == sorted(["file1.txt", "file1.txt", "file3.txt"])

    # Check total amounts
    totals = {
        (r["file__path"], r["file__amount"]): (r["total"], r["cnt"]) for r in records
    }
    assert totals == {
        ("file1.txt", 10): (20, 2),  # 10 + 10 (grouped)
        ("file1.txt", 20): (20, 1),
        ("file3.txt", 40): (40, 1),
    }
