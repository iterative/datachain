import os
from textwrap import dedent

import pytest

from datachain.utils import (
    datachain_paths_join,
    determine_processes,
    import_object,
    retry_with_backoff,
    sizeof_fmt,
    sql_escape_like,
    suffix_to_number,
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


def test_import_object(tmp_path):
    fname = tmp_path / "foo.py"
    code = """\
        def hello():
            return "Hello!"
    """
    fname.write_text(dedent(code))
    func = import_object(f"{fname}:hello")
    assert func() == "Hello!"


def test_import_object_relative(tmp_path, monkeypatch):
    fname = tmp_path / "foo.py"
    code = """\
        def hello():
            return "Hello!"
    """
    fname.write_text(dedent(code))
    monkeypatch.chdir(tmp_path)
    func = import_object("foo.py:hello")
    assert func() == "Hello!"


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
    "parallel,settings,expected",
    (
        (None, None, False),
        (None, "-1", True),
        (None, "0", False),
        (None, "5", 5),
        (-1, "5", True),
        (0, "5", False),
        (10, "5", 10),
    ),
)
def test_determine_processes(parallel, settings, expected):
    if settings is not None:
        os.environ["DATACHAIN_SETTINGS_PARALLEL"] = settings
    assert determine_processes(parallel) == expected
