import pytest

import datachain as dc
from datachain import func


class Data(dc.DataModel):
    path: str


@pytest.fixture
def path_ds(test_session) -> dc.DataChain:
    return dc.read_values(
        id=(1, 2, 3, 4),
        data=(
            Data(path="dir1/dir2/file.txt"),
            Data(path="dir1/dir2/file.tar.gz"),
            Data(path="dir1/dir2/file"),
            Data(path="file.txt"),
        ),
        file=(
            "dir1/dir2/.file.txt",
            "dir1/dir2/file.tar.gz",
            "dir1/dir2/file",
            ".file.txt",
        ),
        session=test_session,
    )


def test_path_file_ext(path_ds):
    ds = list(
        path_ds.mutate(
            t1=func.file_ext("data.path"),
            t2=func.file_ext(dc.C("file")),
            t3=func.file_ext(dc.func.literal("path/to/file.txt")),
            t4=func.file_ext(dc.func.literal("path/to/file")),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert ds == [
        ("txt", "txt", "txt", ""),
        ("gz", "gz", "txt", ""),
        ("", "", "txt", ""),
        ("txt", "txt", "txt", ""),
    ]


def test_path_file_stem(path_ds):
    ds = list(
        path_ds.mutate(
            t1=func.file_stem(dc.C("data.path")),
            t2=func.file_stem("file"),
            t3=func.file_stem(dc.func.literal("path/to/file.txt")),
            t4=func.file_stem(dc.func.literal("path/to/file")),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert ds == [
        ("file", ".file", "file", "file"),
        ("file.tar", "file.tar", "file", "file"),
        ("file", "file", "file", "file"),
        ("file", ".file", "file", "file"),
    ]


def test_path_name(path_ds):
    ds = list(
        path_ds.mutate(
            t1=func.name("data.path"),
            t2=func.name("file"),
            t3=func.name(dc.func.literal("path/to/file.txt")),
            t4=func.name(dc.func.literal("path/to/file")),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert ds == [
        ("file.txt", ".file.txt", "file.txt", "file"),
        ("file.tar.gz", "file.tar.gz", "file.txt", "file"),
        ("file", "file", "file.txt", "file"),
        ("file.txt", ".file.txt", "file.txt", "file"),
    ]


def test_path_parent(path_ds):
    ds = list(
        path_ds.mutate(
            t1=func.parent(dc.C("data.path")),
            t2=func.parent(dc.C("file")),
            t3=func.parent(dc.func.literal("path/to/file.txt")),
            t4=func.parent(dc.func.literal("path/to/file")),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert ds == [
        ("dir1/dir2", "dir1/dir2", "path/to", "path/to"),
        ("dir1/dir2", "dir1/dir2", "path/to", "path/to"),
        ("dir1/dir2", "dir1/dir2", "path/to", "path/to"),
        ("", "", "path/to", "path/to"),
    ]
