import pytest

from datachain import func
from datachain.lib.utils import DataChainParamsError
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


@pytest.mark.parametrize(
    "val,expected",
    [
        (1, "A"),
        (2, "D"),
        (3, "B"),
        (4, "D"),
        (5, "C"),
        (100, "D"),
    ],
)
def test_case(warehouse, val, expected):
    query = select(
        func.case(*[(val < 2, "A"), (2 < val < 4, "B"), (4 < val < 6, "C")], else_="D")
    )
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "val,expected",
    [
        (1, "A"),
        (2, None),
    ],
)
def test_case_without_else(warehouse, val, expected):
    query = select(func.case(*[(val < 2, "A")]))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


def test_case_missing_statements(warehouse):
    with pytest.raises(DataChainParamsError) as exc_info:
        select(func.case(*[], else_="D"))
    assert str(exc_info.value) == "Missing statements"


def test_case_not_same_result_types(warehouse):
    val = 2
    with pytest.raises(DataChainParamsError) as exc_info:
        select(func.case(*[(val > 1, "A"), (2 < val < 4, 5)], else_="D"))
    assert str(exc_info.value) == (
        "Statement values must be of the same type, got <class 'str'> and <class 'int'>"
    )


def test_case_wrong_result_type(warehouse):
    val = 2
    with pytest.raises(DataChainParamsError) as exc_info:
        select(func.case(*[(val > 1, ["a", "b"]), (2 < val < 4, [])], else_=[]))
    assert str(exc_info.value) == (
        "Only python literals ([<class 'int'>, <class 'float'>, "
        "<class 'complex'>, <class 'str'>, <class 'bool'>]) are supported for values"
    )


@pytest.mark.parametrize(
    "val,expected",
    [
        (1, "L"),
        (2, "L"),
        (3, "L"),
        (4, "H"),
        (5, "H"),
        (100, "H"),
    ],
)
def test_ifelse(warehouse, val, expected):
    query = select(func.ifelse(val <= 3, "L", "H"))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "val,expected",
    [
        [None, True],
        [func.literal("abcd"), False],
    ],
)
def test_isnone(warehouse, val, expected):
    from datachain.func.conditional import isnone

    query = select(isnone(val))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "val1,val2,expected",
    [
        [None, func.literal("a"), True],
        [None, None, True],
        [func.literal("a"), func.literal("a"), False],
    ],
)
def test_or(warehouse, val1, val2, expected):
    from datachain.func.conditional import isnone, or_

    query = select(or_(isnone(val1), isnone(val2)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)


@pytest.mark.parametrize(
    "val1,val2,expected",
    [
        [None, func.literal("a"), False],
        [None, None, True],
        [func.literal("a"), func.literal("a"), False],
    ],
)
def test_and(warehouse, val1, val2, expected):
    from datachain.func.conditional import and_, isnone

    query = select(and_(isnone(val1), isnone(val2)))
    result = tuple(warehouse.db.execute(query))
    assert result == ((expected,),)
