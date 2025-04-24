import datachain as dc
from datachain import func


def test_array_first():
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
            first_i=func.array.first("arr.i"),
            first_f=func.array.first("arr.f"),
            first_s=func.array.first(["a", "b", "c", "d"]),
        )
        .collect("first_i", "first_f", "first_s")
    )

    assert set(ds) == {
        (10, 1.0, "a"),
        (40, 4.0, "a"),
    }
