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


@pytest.mark.parametrize(
    "input_string,pattern,replacement,expected",
    [
        ("hello world", "world", "universe", "hello universe"),
        ("abc123def456", r"\d+", "X", "abcXdefX"),
        ("cat.1001.jpg", r"\.(\w+)\.", r"_\1_", "cat_1001_jpg"),
        (
            "dog_photo.jpg",
            r"(\w+)\.(jpg|jpeg|png|gif)$",
            r"\1_thumb.\2",
            "dog_photo_thumb.jpg",
        ),
        ("file.with...dots.txt", r"\.+", ".", "file.with.dots.txt"),
    ],
)
def test_regexp_replace(warehouse, input_string, pattern, replacement, expected):
    query = select(
        string.regexp_replace(
            literal(input_string), literal(pattern), literal(replacement)
        )
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)
