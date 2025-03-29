import os

import pytest

from datachain.utils import (
    datachain_paths_join,
    determine_processes,
    determine_workers,
    nested_dict_path_set,
    retry_with_backoff,
    row_to_nested_dict,
    sizeof_fmt,
    sql_escape_like,
    suffix_to_number,
    uses_glob,
)

DATACHAIN_TEST_PATHS = ["/file1", "file2", "/dir/file3", "dir/file4"]
DATACHAIN_EX_ROOT = ["/file1", "/file2", "/dir/file3", "/dir/file4"]
DATACHAIN_EX_SUBDIR = [
    "subdir/file1",
    "subdir/file2",
    "subdir/dir/file3",
    "subdir/dir/file4",
]
DATACHAIN_EX_DOUBLE_SUBDIR = [
    "subdir/double/file1",
    "subdir/double/file2",
    "subdir/double/dir/file3",
    "subdir/double/dir/file4",
]


@pytest.mark.parametrize(
    "src,paths,expected",
    (
        ("", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_ROOT),
        ("subdir", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_SUBDIR),
        ("subdir/double", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
        ("subdir/double/file*", DATACHAIN_TEST_PATHS, DATACHAIN_EX_DOUBLE_SUBDIR),
    ),
)
def test_datachain_paths_join(src, paths, expected):
    assert list(datachain_paths_join(src, paths)) == expected


@pytest.mark.parametrize(
    "num,suffix,si,expected",
    (
        (1, "", False, "   1"),
        (536, "", False, " 536"),
        (1000, "", False, "1000"),
        (1000, "", True, "1.0K"),
        (1000, " tests", False, "1000 tests"),
        (1000, " tests", True, "1.0K tests"),
        (100000, "", False, "97.7K"),
        (100000, "", True, "100.0K"),
        (1000000, "", True, "1.0M"),
        (1000000000, "", True, "1.0G"),
        (1000000000000, "", True, "1.0T"),
        (1000000000000000, "", True, "1.0P"),
        (1000000000000000000, "", True, "1.0E"),
        (1000000000000000000000, "", True, "1.0Z"),
        (1000000000000000000000000, "", True, "1.0Y"),
        (1000000000000000000000000000, "", True, "1.0R"),
        (1000000000000000000000000000000, "", True, "1.0Q"),
    ),
)
def test_sizeof_fmt(num, suffix, si, expected):
    assert sizeof_fmt(num, suffix, si) == expected


@pytest.mark.parametrize(
    "text,expected",
    (
        ("1", 1),
        ("50", 50),
        ("1K", 1024),
        ("1k", 1024),
        ("2M", 1024 * 1024 * 2),
    ),
)
def test_suffix_to_number(text, expected):
    assert suffix_to_number(text) == expected


@pytest.mark.parametrize(
    "text",
    (
        "",
        "Bogus",
        "50H",
    ),
)
def test_suffix_to_number_invalid(text):
    with pytest.raises(ValueError):
        suffix_to_number(text)


@pytest.mark.parametrize(
    "text,expected",
    (
        ("test like", "test like"),
        ("Can%t \\escape_this", "Can\\%t \\\\escape\\_this"),
    ),
)
def test_sql_escape_like(text, expected):
    assert sql_escape_like(text) == expected


def test_retry_with_backoff():
    called = 0
    retries = 2

    @retry_with_backoff(retries=retries, backoff_sec=0.05)
    def func_with_exception():
        nonlocal called
        called += 1
        raise RuntimeError("Error")

    with pytest.raises(RuntimeError):
        func_with_exception()
    assert called == retries + 1

    called = 0  # resetting called

    @retry_with_backoff(retries=retries, backoff_sec=0.05)
    def func_ok():
        nonlocal called
        called += 1

    func_ok()
    assert called == 1


@pytest.mark.parametrize(
    "workers,rows_total,settings,expected",
    (
        (False, None, None, False),
        (False, None, "-1", False),
        (False, None, "0", False),
        (False, None, "5", 5),
        (-1, None, "5", False),
        (0, None, "5", False),
        (10, None, "5", 10),
        (True, None, None, True),
        (True, None, "5", True),
        (10, 0, None, False),
        (10, 1, None, False),
        (10, 2, None, 10),
        (True, 0, None, False),
        (True, 1, None, False),
        (True, 2, None, True),
    ),
)
def test_determine_workers(workers, rows_total, settings, expected):
    if settings is not None:
        os.environ["DATACHAIN_DISTRIBUTED"] = "some_defined_value"
        os.environ["DATACHAIN_SETTINGS_WORKERS"] = settings
    assert determine_workers(workers, rows_total=rows_total) == expected


@pytest.mark.parametrize(
    "parallel,rows_total,settings,expected",
    (
        (None, None, None, False),
        (None, None, "-1", True),
        (None, None, "0", False),
        (None, None, "5", 5),
        (-1, None, "5", True),
        (0, None, "5", False),
        (10, None, "5", 10),
        (True, None, None, True),
        (True, None, "5", True),
        (10, 0, None, False),
        (10, 1, None, False),
        (10, 2, None, 10),
        (True, 0, None, False),
        (True, 1, None, False),
        (True, 2, None, True),
    ),
)
def test_determine_processes(parallel, rows_total, settings, expected):
    if settings is not None:
        os.environ["DATACHAIN_SETTINGS_PARALLEL"] = settings
    assert determine_processes(parallel, rows_total=rows_total) == expected


@pytest.mark.parametrize(
    "path,expected",
    (
        ("/dogs", False),
        ("/dogs/", False),
        ("/dogs/*", True),
        ("/home/user/bucket/animals/", False),
        ("/home/user/bucket/animals/*", True),
        ("", False),
        ("*", True),
    ),
)
def test_uses_glob(path, expected):
    assert uses_glob(path) is expected


@pytest.mark.parametrize(
    "data,path,value,expected",
    (
        ({}, ["test"], True, {"test": True}),
        ({"extra": False}, ["test"], True, {"extra": False, "test": True}),
        (
            {"extra": False},
            ["test", "nested"],
            True,
            {"extra": False, "test": {"nested": True}},
        ),
        (
            {"extra": False},
            ["test", "nested", "deep"],
            True,
            {"extra": False, "test": {"nested": {"deep": True}}},
        ),
        (
            {"extra": False, "test": {"test2": 5, "nested": {}}},
            ["test", "nested", "deep"],
            True,
            {"extra": False, "test": {"test2": 5, "nested": {"deep": True}}},
        ),
    ),
)
def test_nested_dict_path_set(data, path, value, expected):
    assert nested_dict_path_set(data, path, value) == expected


@pytest.mark.parametrize(
    "headers,row,expected",
    (
        ([["a"], ["b"]], (3, 7), {"a": 3, "b": 7}),
        ([["a"], ["b", "c"]], (3, 7), {"a": 3, "b": {"c": 7}}),
        (
            [["a", "b"], ["a", "c"], ["d"], ["a", "e", "f"]],
            (1, 5, "test", 11),
            {"a": {"b": 1, "c": 5, "e": {"f": 11}}, "d": "test"},
        ),
    ),
)
def test_row_to_nested_dict(headers, row, expected):
    assert row_to_nested_dict(headers, row) == expected
