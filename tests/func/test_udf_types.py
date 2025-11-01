from collections.abc import Iterator

import datachain as dc
from datachain import File


def test_agg_list_file_and_map_count(tmp_dir):
    names = [
        "hotdogs.txt",
        "dogs.txt",
        "dog.txt",
        "1dog.txt",
        "dogatxt.txt",
        "dog.txtx",
    ]

    def collect_files(file: list[File], id: list[int]) -> Iterator[list[File]]:
        # Return the full collection for the partition
        yield file

    def count_files(files: list[File]) -> int:
        return len(files)

    (
        dc.read_values(id=[1] * len(names), file=[File(path=p) for p in names])
        .agg(files=collect_files, partition_by="id")
        .map(num_files=count_files)
        .save("temp_udf_types")
    )

    # Validate result
    ds = dc.read_dataset("temp_udf_types")
    rows = ds.select("num_files").to_list()
    assert rows == [(len(names),)]


def test_agg_list_file_persist_and_read(tmp_dir):
    names = ["a.txt", "b.txt", "c.txt"]

    def collect_files(file: list[File], id: list[int]) -> Iterator[list[File]]:
        yield file

    (
        dc.read_values(id=[1] * len(names), file=[File(path=p) for p in names])
        .agg(files=collect_files, partition_by="id")
        .save("temp_files_only")
    )

    # When reading back, we should get a list of File objects
    ds = dc.read_dataset("temp_files_only")
    vals = ds.select("files").to_list()
    assert len(vals) == 1
    files_list = vals[0][0]
    assert isinstance(files_list, list)
    assert {f.path for f in files_list} == set(names)
