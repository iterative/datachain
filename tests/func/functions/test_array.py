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
            id=(1, 2, 3),
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0], s=["a", "b", "c"]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0], s=["d", "e", "f"]),
                Arr(i=[50], f=[5.0], s=["g"]),
            ),
            ii=(
                [20, 30, 50, 80],
                [10],
                [],
            ),
            ff=(
                [2.0, 3.0, 5.0, 7.0],
                [4.0],
                [],
            ),
            ss=(
                ["b", "c", "e", "f"],
                ["d"],
                [],
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.slice("arr.i", 1),
            t2=func.array.slice(dc.C("arr.i"), 100),
            t3=func.array.slice("arr.f", 0),
            t4=func.array.slice(dc.C("arr.f"), 1, 1),
            t5=func.array.slice(dc.C("arr.s"), 2),
            t6=func.array.slice("arr.s", 1, 10),
            t7=func.array.slice("ii", 1, 2),
            t8=func.array.slice(dc.C("ff"), 5),
            t9=func.array.slice(dc.C("ss"), 1, 5),
            t10=func.array.slice([9.0], 0),
            t11=func.array.slice([17], 5),
            t12=func.array.slice(["a", "b", "c", "d"], 1, 5),
            t13=func.array.slice(["a", "b", "c", "d"], 100),
            t14=func.array.slice([], 0),
        )
        .order_by("id")
        .collect(
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
            "t10",
            "t11",
            "t12",
            "t13",
            "t14",
        )
    )

    assert tuple(ds) == (
        (
            [20, 30],
            [],
            [1.0, 2.0, 3.0],
            [2.0],
            ["c"],
            ["b", "c"],
            [30, 50],
            [],
            ["c", "e", "f"],
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
            [],
            [],
            [],
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
            id=(1, 2, 3),
            arr=(
                Arr(s=["a", "b", "c"]),
                Arr(s=["d"]),
                Arr(s=[]),
            ),
            ss=(
                ["a", "b", "c", "d", "e", "f"],
                ["g"],
                [],
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.join("arr.s", "/"),
            t2=func.array.join(dc.C("arr.s"), ","),
            t3=func.array.join("arr.s"),
            t4=func.array.join("ss", ":"),
            t5=func.array.join(dc.C("ss")),
            t6=func.array.join(["a", "b", "c", "d"], ":"),
            t7=func.array.join(["1", "2"], ","),
            t8=func.array.join([]),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
    )

    assert tuple(ds) == (
        ("a/b/c", "a,b,c", "abc", "a:b:c:d:e:f", "abcdef", "a:b:c:d", "1,2", ""),
        ("d", "d", "d", "g", "g", "a:b:c:d", "1,2", ""),
        ("", "", "", "", "", "a:b:c:d", "1,2", ""),
    )


def test_array_get_element(test_session):
    db_dialect = test_session.catalog.warehouse.db.dialect

    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0]),
                Arr(i=[50], f=[5.0]),
            ),
            ii=(
                [10],
                [50, 60, 70, 80],
                [],
            ),
            ff=(
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 6.0],
                [],
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.get_element("arr.i", 0),
            t2=func.array.get_element(dc.C("arr.i"), 1),
            t3=func.array.get_element("arr.i", 100),
            t4=func.array.get_element("arr.f", 0),
            t5=func.array.get_element("arr.f", 1),
            t6=func.array.get_element(dc.C("ii"), 0),
            t7=func.array.get_element("ff", 3),
            t8=func.array.get_element([9.0], 0),
            t9=func.array.get_element(["a", "b", "c", "d"], 0),
            t10=func.array.get_element(["a", "b", "c", "d"], 1),
            t11=func.array.get_element(["a", "b", "c", "d"], 100),
            t12=func.array.get_element([], 0),
        )
        .order_by("id")
        .collect(
            "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12"
        )
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
                10,
                4.0,
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
                50,
                Float.default_value(db_dialect),
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
                Int.default_value(db_dialect),
                Float.default_value(db_dialect),
                9.0,
                "a",
                "b",
                String.default_value(db_dialect),
                String.default_value(db_dialect),
            ),
        ),
    )


def test_array_length(test_session):
    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]
        s: list[str]

    ds = list(
        dc.read_values(
            id=(1, 2, 3, 4),
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0], s=["a", "b", "c"]),
                Arr(i=[40, 50], f=[4.0, 5.0, 6.0, 7.0], s=["d"]),
                Arr(i=[50], f=[5.0], s=["g"]),
                Arr(i=[], f=[], s=[]),
            ),
            ii=(
                [20, 40, 60, 80],
                [80, 100],
                [130],
                [],
            ),
            ff=(
                [2.0, 4.0, 6.0],
                [8.0, 10.0, 12.0, 14.0],
                [15.0],
                [],
            ),
            ss=(
                ["b", "d", "f", "x", "y", "z"],
                ["h", "j"],
                ["k"],
                [],
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.length("arr.i"),
            t2=func.array.length(dc.C("arr.f")),
            t3=func.array.length("arr.s"),
            t4=func.array.length(dc.C("ii")),
            t5=func.array.length("ff"),
            t6=func.array.length("ss"),
            t7=func.array.length([1, 2, 3, 4, 5]),
            t8=func.array.length([]),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
    )

    assert tuple(ds) == (
        (3, 3, 3, 4, 3, 6, 5, 0),
        (2, 4, 1, 2, 4, 2, 5, 0),
        (1, 1, 1, 1, 1, 1, 5, 0),
        (0, 0, 0, 0, 0, 0, 5, 0),
    )


def test_array_contains(test_session):
    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]
        s: list[str]

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0], s=["a", "b", "c"]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0], s=["d", "e", "f"]),
                Arr(i=[50], f=[5.0], s=["g"]),
            ),
            ii=(
                [20, 30, 50, 80],
                [10],
                [],
            ),
            ff=(
                [2.0, 3.0, 5.0, 7.0],
                [4.0],
                [],
            ),
            ss=(
                ["b", "c", "e", "f"],
                ["d"],
                [],
            ),
            session=test_session,
        )
        .mutate(
            t1=func.array.contains("arr.i", 20),
            t2=func.array.contains(dc.C("arr.i"), 100),
            t3=func.array.contains(dc.C("arr.f"), 2.0),
            t4=func.array.contains("arr.f", 7.0),
            t5=func.array.contains(dc.C("arr.s"), "b"),
            t6=func.array.contains("arr.s", "x"),
            t7=func.array.contains(dc.C("ii"), 30),
            t8=func.array.contains("ii", 100),
            t9=func.array.contains("ff", 5.0),
            t10=func.array.contains(dc.C("ff"), 8.0),
            t11=func.array.contains("ss", "e"),
            t12=func.array.contains(dc.C("ss"), "z"),
            t13=func.array.contains([1, 2, 3, 4, 5], 3),
            t14=func.array.contains([1, 2, 3, 4, 5], 7),
            t15=func.array.contains([], 1),
        )
        .order_by("id")
        .collect(
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
            "t10",
            "t11",
            "t12",
            "t13",
            "t14",
            "t15",
        )
    )

    assert ds == [
        (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ]
