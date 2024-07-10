import pytest

from datachain.sql import literal, select
from datachain.sql.functions import string


def test_length(warehouse):
    query = select(string.length(literal("abcdefg")))
    result = tuple(warehouse.db.execute(query))
    assert result == ((7,),)


@pytest.mark.parametrize(
    "args,expected",
    [
        ([literal("abc//def/g/hi"), literal("/")], ["abc", "", "def", "g", "hi"]),
        ([literal("abc//def/g/hi"), literal("/"), 2], ["abc", "", "def/g/hi"]),
    ],
)
def test_split(warehouse, args, expected):
    query = select(string.split(*args))
    result = tuple(warehouse.dataset_rows_select(query))
    assert result == ((expected,),)
