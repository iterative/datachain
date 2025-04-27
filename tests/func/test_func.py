import datachain as dc
from datachain import func


def test_array_get_element():
    class Arr(dc.DataModel):
        i: list[int]
        f: list[float]

    ds = (
        dc.read_values(
            arr=(
                Arr(i=[10, 20, 30], f=[1.0, 2.0, 3.0]),
                Arr(i=[40, 50, 60], f=[4.0, 5.0, 6.0]),
            ),
        )
        .mutate(
            first_i=func.array.get_element("arr.i", 0),
            second_i=func.array.get_element("arr.i", 1),
            unknown_i=func.array.get_element("arr.i", 100),
            first_f=func.array.get_element("arr.f", 0),
            second_f=func.array.get_element("arr.f", 1),
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
            "first_s",
            "second_s",
            "unknown_s",
            "unknown",
        )
    )

    assert set(ds) == {
        (10, 20, None, 1.0, 2.0, "a", "b", None, None),
        (40, 50, None, 4.0, 5.0, "a", "b", None, None),
    }
