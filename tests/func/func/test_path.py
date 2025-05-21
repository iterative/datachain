import datachain as dc
from datachain import func


def test_path_functions(test_session):
    class Data(dc.DataModel):
        path: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(path="dir1/dir2/file.txt"),
                Data(path="dir1/dir2/file.tar.gz"),
                Data(path="file.txt"),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.file_ext("data.path"),
            t2=func.file_stem("data.path"),
            t3=func.name("data.path"),
            t4=func.parent("data.path"),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert tuple(ds) == (
        ("txt", "file", "file.txt", "dir1/dir2"),
        ("gz", "file.tar", "file.tar.gz", "dir1/dir2"),
        ("txt", "file", "file.txt", ""),
    )


def test_path_functions_with_dots(test_session):
    class Data(dc.DataModel):
        path: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(path=".file.txt"),
                Data(path="dir1/.file.txt"),
                Data(path="dir1/dir2/.file.txt"),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.file_ext("data.path"),
            t2=func.file_stem("data.path"),
            t3=func.name("data.path"),
            t4=func.parent("data.path"),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert tuple(ds) == (
        ("txt", ".file", ".file.txt", ""),
        ("txt", ".file", ".file.txt", "dir1"),
        ("txt", ".file", ".file.txt", "dir1/dir2"),
    )


def test_path_functions_with_multiple_extensions(test_session):
    class Data(dc.DataModel):
        path: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(path="file.txt.gz"),
                Data(path="file.tar.gz"),
                Data(path="file.min.js"),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.file_ext("data.path"),
            t2=func.file_stem("data.path"),
            t3=func.name("data.path"),
            t4=func.parent("data.path"),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4")
    )

    assert tuple(ds) == (
        ("gz", "file.txt", "file.txt.gz", ""),
        ("gz", "file.tar", "file.tar.gz", ""),
        ("js", "file.min", "file.min.js", ""),
    )
