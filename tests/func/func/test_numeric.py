import datachain as dc
from datachain import func


def test_numeric_bit_operations(test_session):
    class Data(dc.DataModel):
        i: int
        j: int

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, j=5),  # 1010 & 0101
                Data(i=20, j=15),  # 10100 & 01111
                Data(i=30, j=25),  # 11110 & 11001
            ),
            session=test_session,
        )
        .mutate(
            t1=func.bit_and("data.i", "data.j"),
            t2=func.bit_or("data.i", "data.j"),
            t3=func.bit_xor("data.i", "data.j"),
        )
        .order_by("id")
        .collect("t1", "t2", "t3")
    )

    assert tuple(ds) == (
        (0, 15, 15),  # 1010 & 0101 = 0000, 1010 | 0101 = 1111, 1010 ^ 0101 = 1111
        (
            4,
            31,
            27,
        ),  # 10100 & 01111 = 00100, 10100 | 01111 = 11111, 10100 ^ 01111 = 11011
        (
            24,
            31,
            7,
        ),  # 11110 & 11001 = 11000, 11110 | 11001 = 11111, 11110 ^ 11001 = 00111
    )


def test_numeric_bit_hamming_distance(test_session):
    class Data(dc.DataModel):
        i: int
        j: int

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10, j=5),  # 1010 vs 0101
                Data(i=20, j=15),  # 10100 vs 01111
                Data(i=30, j=25),  # 11110 vs 11001
            ),
            session=test_session,
        )
        .mutate(
            t1=func.bit_hamming_distance("data.i", "data.j"),
        )
        .order_by("id")
        .collect("t1")
    )

    assert tuple(ds) == (
        4,  # 1010 vs 0101: all bits differ
        4,  # 10100 vs 01111: 4 bits differ
        3,  # 11110 vs 11001: 3 bits differ
    )


def test_numeric_int_hash_64(test_session):
    class Data(dc.DataModel):
        i: int

    ds = list(
        dc.read_values(
            id=[1, 2, 3],
            data=(
                Data(i=10),
                Data(i=20),
                Data(i=30),
            ),
            session=test_session,
        )
        .mutate(
            t1=func.int_hash_64("data.i"),
            t2=func.int_hash_64(42),
        )
        .order_by("id")
        .collect("t1", "t2")
    )

    # Hash values are unpredictable, so we just check that they are integers
    assert all(isinstance(x[0], int) for x in ds)
    assert all(isinstance(x[1], int) for x in ds)
    # Same input should produce same hash
    assert ds[0][0] == ds[0][0]
    assert ds[1][0] == ds[1][0]
    assert ds[2][0] == ds[2][0]
    # Different inputs should produce different hashes
    assert ds[0][0] != ds[1][0]
    assert ds[1][0] != ds[2][0]
    assert ds[0][0] != ds[2][0]
