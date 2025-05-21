import datachain as dc
from datachain import func


def test_aggregate_avg(test_session):
    class Data(dc.DataModel):
        i: int
        f: float

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, f=1.0),
                Data(i=20, f=2.0),
                Data(i=30, f=3.0),
            ),
            session=test_session,
        )
        .group_by(
            t1=func.avg("data.i"),
            t2=func.avg("data.f"),
        )
        .collect("t1", "t2")
    )

    assert tuple(ds) == ((20.0, 2.0),)


def test_aggregate_count(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            session=test_session,
        )
        .group_by(
            t1=func.count("data.i"),
            t2=func.count("data.f"),
            t3=func.count("data.s"),
        )
        .collect("t1", "t2", "t3")
    )

    assert tuple(ds) == ((3, 3, 3),)


def test_aggregate_sum(test_session):
    class Data(dc.DataModel):
        i: int
        f: float

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, f=1.0),
                Data(i=20, f=2.0),
                Data(i=30, f=3.0),
            ),
            session=test_session,
        )
        .group_by(
            t1=func.sum("data.i"),
            t2=func.sum("data.f"),
        )
        .collect("t1", "t2")
    )

    assert tuple(ds) == ((60, 6.0),)


def test_aggregate_min_max(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            session=test_session,
        )
        .group_by(
            t1=func.min("data.i"),
            t2=func.min("data.f"),
            t3=func.min("data.s"),
            t4=func.max("data.i"),
            t5=func.max("data.f"),
            t6=func.max("data.s"),
        )
        .collect("t1", "t2", "t3", "t4", "t5", "t6")
    )

    assert tuple(ds) == ((10, 1.0, "a", 30, 3.0, "c"),)


def test_aggregate_any_value(test_session):
    class Data(dc.DataModel):
        i: int
        f: float
        s: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, f=1.0, s="a"),
                Data(i=20, f=2.0, s="b"),
                Data(i=30, f=3.0, s="c"),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.any_value("data.i"),
            t2=func.any_value("data.f"),
            t3=func.any_value("data.s"),
        )
        .order_by("id")
        .collect("t1", "t2", "t3")
    )

    # any_value can return any value from the group,
    # so we just check that it returns a value
    assert all(isinstance(x[0], int) for x in ds)
    assert all(isinstance(x[1], float) for x in ds)
    assert all(isinstance(x[2], str) for x in ds)
