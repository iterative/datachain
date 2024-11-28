import math

import pytest

from datachain import func
from datachain.sql import select


def test_cosine_distance(warehouse):
    query = select(
        func.cosine_distance((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)).label("cos1"),
        func.cosine_distance([3.0, 5.0, 1.0], (3.0, 5.0, 1.0)).label("cos2"),
        func.cosine_distance((1, 0), [0, 10]).label("cos3"),
        func.cosine_distance([0.0, 10.0], [1.0, 0.0]).label("cos4"),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((0.0, 0.0, 1.0, 1.0),)


def test_euclidean_distance(warehouse):
    query = select(
        func.euclidean_distance((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)).label("eu1"),
        func.euclidean_distance([3.0, 5.0, 1.0], (3.0, 5.0, 1.0)).label("eu2"),
        func.euclidean_distance((1, 0), [0, 1]).label("eu3"),
        func.euclidean_distance([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]).label("eu4"),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((0.0, 0.0, math.sqrt(2), math.sqrt(3)),)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["signal"],
        [[1, 2]],
        [[1, 2], [1, 2], [1, 2]],
        ["signal1", "signal2", "signal3"],
        ["signal1", "signal2", [1, 2]],
    ],
)
def test_cosine_euclidean_distance_error_args(warehouse, args):
    with pytest.raises(ValueError, match="requires exactly two arguments"):
        func.cosine_distance(*args)

    with pytest.raises(ValueError, match="requires exactly two arguments"):
        func.euclidean_distance(*args)


def test_cosine_euclidean_distance_error_vectors_length(warehouse):
    with pytest.raises(ValueError, match="requires vectors of the same length"):
        func.cosine_distance([1], [1, 2])

    with pytest.raises(ValueError, match="requires vectors of the same length"):
        func.euclidean_distance([1], [1, 2])


def test_length(warehouse):
    query = select(
        func.length(["abc", "def", "g", "hi"]).label("len1"),
        func.length([3.0, 5.0, 1.0, 6.0, 1.0]).label("len2"),
        func.length([[1, 2, 3], [4, 5, 6]]).label("len3"),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((4, 5, 2),)


def test_length_on_split(warehouse):
    query = select(
        func.array.length(func.string.split(func.literal("abc/def/g/hi"), "/")),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((4,),)
