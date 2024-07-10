from datachain.sql import literal, select
from datachain.sql.functions import array, string


def test_length(warehouse):
    query = select(
        array.length(["abc", "def", "g", "hi"]),
        array.length([3.0, 5.0, 1.0, 6.0, 1.0]),
        array.length([[1, 2, 3], [4, 5, 6]]),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((4, 5, 2),)


def test_length_on_split(warehouse):
    query = select(
        array.length(string.split(literal("abc/def/g/hi"), literal("/"))),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((4,),)
