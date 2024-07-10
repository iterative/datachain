import pytest

from datachain.sql.sqlite.base import functions_exist


@pytest.mark.parametrize(
    "names,expected",
    [
        (["sum", "abs", "upper"], True),
        (["sum", "abs", "upper", "missing_func"], False),
        ([], True),
    ],
)
def test_functions_exist(names, expected):
    assert functions_exist(names) == expected
