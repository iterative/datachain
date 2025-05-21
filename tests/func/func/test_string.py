import datachain as dc
from datachain import func


def test_string_byte_hamming_distance(test_session):
    class Data(dc.DataModel):
        s1: str
        s2: str

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(s1="hello", s2="world"),  # Different strings
                Data(s1="hello", s2="hello"),  # Same strings
                Data(s1="", s2=""),  # Empty strings
            ),
            session=test_session,
        )
        .mutate(
            t1=func.byte_hamming_distance("data.s1", "data.s2"),
        )
        .order_by("id")
        .collect("t1")
    )

    assert tuple(ds) == (
        4,  # "hello" vs "world": 4 bytes differ
        0,  # "hello" vs "hello": 0 bytes differ
        0,  # "" vs "": 0 bytes differ
    )
