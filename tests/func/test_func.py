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


def tuples_almost_equal(t1, t2):
    """Compare two tuples, treating NaN floats as equal."""
    if len(t1) != len(t2):
        return False
    return all(values_almost_equal(x, y) for x, y in zip(t1, t2))


def sets_of_tuples_almost_equal(s1, s2):
    """Compare two sets of tuples, treating NaN floats as equal."""
    if len(s1) != len(s2):
        return False
    unmatched = list(s2)
    for item1 in s1:
        for item2 in unmatched:
            if tuples_almost_equal(item1, item2):
                unmatched.remove(item2)
                break
        else:
            return False
    return True


def test_array_get_element(test_session):
    db_dialect = test_session.catalog.warehouse.db.dialect

    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]

    ds = (
        dc.read_values(
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0]),
                Arr(i=[50], f=[5.0]),
            ),
            session=test_session,
        )
        .mutate(
            first_i=func.array.get_element("arr.i", 0),
            second_i=func.array.get_element("arr.i", 1),
            unknown_i=func.array.get_element("arr.i", 100),
            first_f=func.array.get_element("arr.f", 0),
            second_f=func.array.get_element("arr.f", 1),
            first_f2=func.array.get_element([9.0], 0),
            first_s=func.array.get_element(["a", "b", "c", "d"], 0),
            second_s=func.array.get_element(["a", "b", "c", "d"], 1),
            unknown_s=func.array.get_element(["a", "b", "c", "d"], 100),
            unknown=func.array.get_element([], 0),
        )
        .collect(
            "first_i",
            "second_i",
            "unknown_i",
            "first_f",
            "second_f",
            "first_f2",
            "first_s",
            "second_s",
            "unknown_s",
            "unknown",
        )
    )

    assert sets_of_tuples_almost_equal(
        set(ds),
        {
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
        },
    )
