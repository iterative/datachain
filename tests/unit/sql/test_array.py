import math

import pytest
from numpy.testing import assert_array_almost_equal

from datachain import func
from datachain.sql import select
from datachain.sql.types import Int, String


def test_cosine_distance(warehouse):
    query = select(
        func.cosine_distance((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)).label("cos1"),
        func.cosine_distance([3.0, 5.0, 1.0], (3.0, 5.0, 1.0)).label("cos2"),
        func.cosine_distance((1, 0), [0, 10]).label("cos3"),
        func.cosine_distance([0.0, 10.0], [1.0, 0.0]).label("cos4"),
    )
    result = tuple(warehouse.db.execute(query))
    assert len(result) == 1
    assert_array_almost_equal(result[0], (0.0, 0.0, 1.0, 1.0), decimal=6)


def test_euclidean_distance(warehouse):
    query = select(
        func.euclidean_distance((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)).label("eu1"),
        func.euclidean_distance([3.0, 5.0, 1.0], (3.0, 5.0, 1.0)).label("eu2"),
        func.euclidean_distance((1, 0), [0, 1]).label("eu3"),
        func.euclidean_distance([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]).label("eu4"),
    )
    result = tuple(warehouse.db.execute(query))
    assert len(result) == 1
    assert_array_almost_equal(
        result[0], (0.0, 0.0, math.sqrt(2), math.sqrt(3)), decimal=6
    )


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


def test_get_element(warehouse):
    db_dialect = warehouse.db.dialect

    query = select(
        func.array.get_element(["abc", "def", "g", "hi"], 0).label("first1"),
        func.array.get_element(["abc", "def", "g", "hi"], 1).label("second1"),
        func.array.get_element([3.0, 5.0, 1.0, 6.0, 1.0], 0).label("first2"),
        func.array.get_element([3.0, 5.0, 1.0, 6.0, 1.0], 1).label("second2"),
        func.array.get_element([1, 2, 3, 4, 5, 6], 0).label("first3"),
        func.array.get_element([1, 2, 3, 4, 5, 6], 1).label("second3"),
        func.array.get_element([1], 0).label("first4"),
        func.array.get_element([2.0], 0).label("first5"),
        func.array.get_element([], 0).label("not_found1"),
        func.array.get_element([], -1).label("not_found2"),
        func.array.get_element([1], 2).label("not_found3"),
    )
    result = tuple(warehouse.dataset_rows_select(query))
    assert result == (
        (
            "abc",
            "def",
            3.0,
            5.0,
            1,
            2,
            1,
            2.0,
            String.default_value(db_dialect),
            String.default_value(db_dialect),
            Int.default_value(db_dialect),
        ),
    )


def test_contains(warehouse):
    query = select(
        func.contains(["abc", "def", "g", "hi"], "abc").label("contains1"),
        func.contains(["abc", "def", "g", "hi"], "cdf").label("contains2"),
        func.contains([3.0, 5.0, 1.0, 6.0, 1.0], 1.0).label("contains3"),
        func.contains([[1, None, 3], [4, 5, 6]], [1, None, 3]).label("contains4"),
        # Not supported yet by CH, need to add it later + some Pydantic model as
        # an input:
        # func.contains(
        #     [{"c": 1, "a": True}, {"b": False}], {"a": True, "c": 1}
        # ).label("contains5"),
        func.contains([1, None, 3], None).label("contains6"),
        func.contains([1, True, 3], True).label("contains7"),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((1, 0, 1, 1, 1, 1),)


def test_slice(warehouse):
    query = select(
        func.array.slice(["abc", "def", "g", "hi"], 1).label("slice1"),
        func.array.slice(["abc", "def", "g", "hi"], 0, 10).label("slice2"),
        func.array.slice(["abc", "def", "g", "hi"], 0, 2).label("slice3"),
        func.array.slice([3.0, 5.0, 1.0, 6.0, 1.0], 4).label("slice4"),
        func.array.slice([[1, None, 3], [4, 5, 6]], 0).label("slice5"),
        func.array.slice([1, None, 3], 1, 10).label("slice6"),
        func.array.slice([1, True, 3], 0).label("slice7"),
    )
    result = tuple(warehouse.dataset_rows_select(query))
    assert result == (
        (
            ["def", "g", "hi"],
            ["abc", "def", "g", "hi"],
            ["abc", "def"],
            [1.0],
            [[1, None, 3], [4, 5, 6]],
            [None, 3],
            [1, True, 3],
        ),
    )


def test_join(warehouse):
    query = select(
        func.array.join(["abc", "def", "g", "hi"], "/").label("join1"),
        func.array.join(["abc", "def", "g", "hi"], ",").label("join2"),
        func.array.join(["abc", "def", "g", "hi"]).label("join3"),
        func.array.join(["abc"], ":").label("join4"),
    )
    result = tuple(warehouse.dataset_rows_select(query))
    assert result == (
        (
            "abc/def/g/hi",
            "abc,def,g,hi",
            "abcdefghi",
            "abc",
        ),
    )


def test_length_on_split(warehouse):
    query = select(
        func.array.length(func.string.split(func.literal("abc/def/g/hi"), "/")),
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((4,),)
