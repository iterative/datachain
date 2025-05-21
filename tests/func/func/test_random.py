import datachain as dc
from datachain import func


def test_random_rand(test_session):
    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            session=test_session,
        )
        .mutate(
            t1=func.rand(),
        )
        .order_by("id")
        .collect("t1")
    )

    # Check that we got 3 random integers
    assert len(ds) == 3
    assert all(isinstance(x, int) for x in ds)
    # Check that we got different random numbers
    assert len(set(ds)) == 3
