import pytest

from datachain import func
from datachain.sql import select, values


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            [
                func.literal("abc"),
                func.literal("bcd"),
                func.literal("Abc"),
                func.literal("cd"),
            ],
            "cd",
        ),
        ([3, 1, 2.0, 3.1, 2.5, -1], 3.1),
        ([4], 4),
    ],
)
def test_greatest(warehouse, args, expected):
    query = select(func.greatest(*args))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "args,expected",
    [
        (
            [
                func.literal("abc"),
                func.literal("bcd"),
                func.literal("Abc"),
                func.literal("cd"),
            ],
            "Abc",
        ),
        ([3, 1, 2.0, 3.1, 2.5, -1], -1),
        ([4], 4),
    ],
)
def test_least(warehouse, args, expected):
    query = select(func.least(*args))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "expr,expected",
    [
        (func.greatest("a"), [(3,), (8,), (9,)]),
        (func.least("a"), [(3,), (8,), (9,)]),
        (func.least("a", "b"), [(3,), (7,), (1,)]),
    ],
)
def test_conditionals_with_multiple_rows(warehouse, expr, expected):
    # In particular, we want to ensure that we are avoiding sqlite's
    # default behavior for `max` and `min` which is to behave as
    # aggregate functions when a single argument is passed.
    # See https://www.sqlite.org/lang_corefunc.html#max_scalar
    query = select(expr).select_from(values([(3, 5), (8, 7), (9, 1)], ["a", "b"]))
    result = list(warehouse.db.execute(query))
    assert result == expected
