import math

import datachain as dc
from datachain import func
from datachain.sql.types import Float, Int, String


def values_almost_equal(a, b):
    """Compare two values, treating NaNs as equal."""
    if (
        isinstance(a, float)
        and isinstance(b, float)
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    return a == b


def tuples_almost_equal(t1, t2, path=""):
    """Compare two tuples, treating NaN floats as equal."""
    if len(t1) != len(t2):
        raise AssertionError(
            f"Tuple length mismatch at {path or 'root'}: {len(t1)} != {len(t2)}\n"
            f"  Left ({type(t1)}): {t1}\n"
            f"  Right ({type(t2)}): {t2}"
        )

    for i, (x, y) in enumerate(zip(t1, t2)):
        subpath = f"{path}[{i}]"
        if isinstance(x, tuple) and isinstance(y, tuple):
            tuples_almost_equal(x, y, path=subpath)
        elif not values_almost_equal(x, y):
            raise AssertionError(
                f"Mismatch at {subpath}:\n"
                f"  Left ({type(x)}): {x}\n"
                f"  Right ({type(y)}): {y}"
            )


def test_array_slice(test_session):
    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]
        s: list[str]

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0], s=["a", "b", "c"]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0], s=["d", "e", "f"]),
                Arr(i=[50], f=[5.0], s=["g"]),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.slice("arr.i", 1),
            t2=func.array.slice("arr.i", 100),
            t3=func.array.slice("arr.f", 0),
            t4=func.array.slice("arr.f", 1, 1),
            t5=func.array.slice("arr.s", 2),
            t6=func.array.slice("arr.s", 1, 10),
            t7=func.array.slice([9.0], 0),
            t8=func.array.slice([17], 5),
            t9=func.array.slice(["a", "b", "c", "d"], 1, 5),
            t10=func.array.slice(["a", "b", "c", "d"], 100),
            t11=func.array.slice([], 0),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11")
    )

    assert tuple(ds) == (
        (
            [20, 30],
            [],
            [1.0, 2.0, 3.0],
            [2.0],
            ["c"],
            ["b", "c"],
            [9.0],
            [],
            ["b", "c", "d"],
            [],
            [],
        ),
        (
            [50, 60],
            [],
            [4.0, 5.0, 6.0],
            [5.0],
            ["f"],
            ["e", "f"],
            [9.0],
            [],
            ["b", "c", "d"],
            [],
            [],
        ),
        (
            [],
            [],
            [5.0],
            [],
            [],
            [],
            [9.0],
            [],
            ["b", "c", "d"],
            [],
            [],
        ),
    )


def test_array_join(test_session):
    class Arr(dc.DataModel):
        s: list[str]

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            arr=(
                Arr(s=["a", "b", "c"]),
                Arr(s=["d"]),
                Arr(s=[]),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.join("arr.s", "/"),
            t2=func.array.join("arr.s", ","),
            t3=func.array.join("arr.s"),
            t4=func.array.join(["a", "b", "c", "d"], ":"),
            t5=func.array.join(["1", "2"], ","),
            t6=func.array.join([]),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6")
    )

    assert tuple(ds) == (
        ("a/b/c", "a,b,c", "abc", "a:b:c:d", "1,2", ""),
        ("d", "d", "d", "a:b:c:d", "1,2", ""),
        ("", "", "", "a:b:c:d", "1,2", ""),
    )


def test_array_get_element(test_session):
    db_dialect = test_session.catalog.warehouse.db.dialect

    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0]),
                Arr(i=[50], f=[5.0]),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.get_element("arr.i", 0),
            t2=func.array.get_element("arr.i", 1),
            t3=func.array.get_element("arr.i", 100),
            t4=func.array.get_element("arr.f", 0),
            t5=func.array.get_element("arr.f", 1),
            t6=func.array.get_element([9.0], 0),
            t7=func.array.get_element(["a", "b", "c", "d"], 0),
            t8=func.array.get_element(["a", "b", "c", "d"], 1),
            t9=func.array.get_element(["a", "b", "c", "d"], 100),
            t10=func.array.get_element([], 0),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10")
    )

    tuples_almost_equal(
        tuple(ds),
        (
            (
                10,
                20,
                Int.default_value(db_dialect),
                1.0,
                2.0,
                9.0,
                "a",
                "b",
                String.default_value(db_dialect),
                String.default_value(db_dialect),
            ),
            (
                40,
                50,
                Int.default_value(db_dialect),
                4.0,
                5.0,
                9.0,
                "a",
                "b",
                String.default_value(db_dialect),
                String.default_value(db_dialect),
            ),
            (
                50,
                Int.default_value(db_dialect),
                Int.default_value(db_dialect),
                5.0,
                Float.default_value(db_dialect),
                9.0,
                "a",
                "b",
                String.default_value(db_dialect),
                String.default_value(db_dialect),
            ),
        ),
    )
