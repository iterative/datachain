import datachain as dc
from datachain import func


def test_numeric_bit_operations(test_session):
    class Data(dc.DataModel):
        i: int
        j: int

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, j=5),
                Data(i=20, j=15),
                Data(i=30, j=25),
            ),
            ii=(20, 40, 60),
            jj=(15, 30, 50),
            session=test_session,
        )
        .mutate(
            t1=func.bit_and("data.i", "data.j"),
            t2=func.bit_or(dc.C("data.i"), "data.j"),
            t3=func.bit_xor("data.i", dc.C("data.j")),
            t4=func.bit_and(dc.C("ii"), dc.C("jj")),
            t5=func.bit_or("ii", 0x0F),
            t6=func.bit_xor(dc.C("ii"), 0x0F),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6")
    )

    assert ds == [
        (0, 15, 15, 4, 31, 27),
        (4, 31, 27, 8, 47, 39),
        (24, 31, 7, 48, 63, 51),
    ]


def test_numeric_bit_hamming_distance(test_session):
    class Data(dc.DataModel):
        i: int
        j: int

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10, j=5),
                Data(i=20, j=15),
                Data(i=30, j=25),
            ),
            ii=(20, 40, 60),
            jj=(15, 30, 50),
            session=test_session,
        )
        .mutate(
            t1=func.bit_hamming_distance("data.i", "data.j"),
            t2=func.bit_hamming_distance(dc.C("data.i"), "data.j"),
            t3=func.bit_hamming_distance("data.i", dc.C("data.j")),
            t4=func.bit_hamming_distance(dc.C("data.i"), dc.C("data.j")),
            t5=func.bit_hamming_distance("ii", dc.C("jj")),
            t6=func.bit_hamming_distance(dc.C("ii"), "jj"),
            t7=func.bit_hamming_distance("ii", 45),
            t8=func.bit_hamming_distance(37, dc.C("data.j")),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8")
    )

    assert ds == [
        (4, 4, 4, 4, 4, 4, 4, 1),
        (4, 4, 4, 4, 4, 4, 2, 3),
        (3, 3, 3, 3, 3, 3, 2, 4),
    ]


def test_numeric_int_hash_64(test_session):
    class Data(dc.DataModel):
        i: int

    ds = list(
        dc.read_values(
            id=(1, 2, 3),
            data=(
                Data(i=10),
                Data(i=20),
                Data(i=30),
            ),
            ii=(20, 40, 60),
            session=test_session,
        )
        .mutate(
            t1=func.int_hash_64("data.i"),
            t2=func.int_hash_64(dc.C("data.i")),
            t3=func.int_hash_64("ii"),
            t4=func.int_hash_64(dc.C("ii")),
            t5=func.int_hash_64(42),
        )
        .order_by("id")
        .collect("t1", "t2", "t3", "t4", "t5")
    )

    assert [tuple(col & (2**64 - 1) for col in row) for row in ds] == [
        (
            8496710636302058981,
            8496710636302058981,
            1626306447464072420,
            1626306447464072420,
            11490350930367293593,
        ),
        (
            1626306447464072420,
            1626306447464072420,
            2973782029276838589,
            2973782029276838589,
            11490350930367293593,
        ),
        (
            637170949039862475,
            637170949039862475,
            11899937643778244071,
            11899937643778244071,
            11490350930367293593,
        ),
    ]
