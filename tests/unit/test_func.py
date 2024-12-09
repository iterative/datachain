import pytest
from sqlalchemy import Label

from datachain import DataChain
from datachain.func import (
    bit_hamming_distance,
    byte_hamming_distance,
    int_hash_64,
    literal,
)
from datachain.func.random import rand
from datachain.func.string import length as strlen
from datachain.lib.signal_schema import SignalSchema
from datachain.sql.sqlite.base import (
    sqlite_bit_hamming_distance,
    sqlite_byte_hamming_distance,
    sqlite_int_hash_64,
)


@pytest.fixture()
def dc():
    return DataChain.from_values(
        num=list(range(1, 6)),
        val=["x" * i for i in range(1, 6)],
    )


def test_db_cols():
    rnd = rand()
    assert rnd._db_cols == []
    assert rnd._db_col_type(SignalSchema({})) is None


def test_label():
    rnd = rand()
    assert rnd.col_label is None
    assert rnd.label("test2") == "test2"

    f = rnd.label("test")
    assert f.col_label == "test"
    assert f.label("test2") == "test2"


def test_col_name():
    rnd = rand()
    assert rnd.get_col_name() == "rand"
    assert rnd.label("test").get_col_name() == "test"
    assert rnd.get_col_name("test2") == "test2"


def test_result_type():
    rnd = rand()
    assert rnd.get_result_type(SignalSchema({})) is int


def test_get_column():
    rnd = rand()
    col = rnd.get_column(SignalSchema({}))
    assert isinstance(col, Label)
    assert col.name == "rand"


def test_add():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 + 1
    assert str(f) == "add()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 + rnd2
    assert str(f) == "add()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 + rnd2
    assert str(f) == "add()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_add_mutate(dc):
    res = dc.mutate(test=strlen("val") + 1).order_by("num").collect("test")
    assert list(res) == [2, 3, 4, 5, 6]

    res = dc.mutate(test=1 + strlen("val")).order_by("num").collect("test")
    assert list(res) == [2, 3, 4, 5, 6]

    res = dc.mutate(test=strlen("val") + strlen("val")).order_by("num").collect("test")
    assert list(res) == [2, 4, 6, 8, 10]


def test_sub():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 - 1
    assert str(f) == "sub()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 - rnd2
    assert str(f) == "sub()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 - rnd2
    assert str(f) == "sub()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_sub_mutate(dc):
    res = dc.mutate(test=strlen("val") - 1).order_by("num").collect("test")
    assert list(res) == [0, 1, 2, 3, 4]

    res = dc.mutate(test=5 - strlen("val")).order_by("num").collect("test")
    assert list(res) == [4, 3, 2, 1, 0]

    res = dc.mutate(test=strlen("val") - strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_mul():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 * 2
    assert str(f) == "mul()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 * rnd2
    assert str(f) == "mul()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 * rnd2
    assert str(f) == "mul()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_mul_mutate(dc):
    res = dc.mutate(test=strlen("val") * 2).order_by("num").collect("test")
    assert list(res) == [2, 4, 6, 8, 10]

    res = dc.mutate(test=3 * strlen("val")).order_by("num").collect("test")
    assert list(res) == [3, 6, 9, 12, 15]

    res = dc.mutate(test=strlen("val") * strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 4, 9, 16, 25]


def test_truediv():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 / 2
    assert str(f) == "div()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 / rnd2
    assert str(f) == "div()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 / rnd2
    assert str(f) == "div()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_truediv_mutate(dc):
    res = dc.mutate(test=strlen("val") / 2).order_by("num").collect("test")
    assert list(res) == [0.5, 1.0, 1.5, 2.0, 2.5]

    res = dc.mutate(test=10 / strlen("val")).order_by("num").collect("test")
    assert list(res) == [10.0, 5.0, 10 / 3, 2.5, 2.0]

    res = dc.mutate(test=strlen("val") / strlen("val")).order_by("num").collect("test")
    assert list(res) == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_floordiv():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 // 2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 1 // rnd2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 // rnd2
    assert str(f) == "floordiv()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_floordiv_mutate(dc):
    res = dc.mutate(test=strlen("val") // 2).order_by("num").collect("test")
    assert list(res) == [0, 1, 1, 2, 2]

    res = dc.mutate(test=10 // strlen("val")).order_by("num").collect("test")
    assert list(res) == [10, 5, 3, 2, 2]

    res = dc.mutate(test=strlen("val") // strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 1, 1]


def test_mod():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 % 2
    assert str(f) == "mod()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 10 % rnd2
    assert str(f) == "mod()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 % rnd2
    assert str(f) == "mod()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_mod_mutate(dc):
    res = dc.mutate(test=strlen("val") % 2).order_by("num").collect("test")
    assert list(res) == [1, 0, 1, 0, 1]

    res = dc.mutate(test=10 % strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 1, 2, 0]

    res = dc.mutate(test=strlen("val") % strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_and():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 & 2
    assert str(f) == "and()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 & rnd2
    assert str(f) == "and()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 & rnd2
    assert str(f) == "and()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_and_mutate(dc):
    res = dc.mutate(test=strlen("val") & 2).order_by("num").collect("test")
    assert list(res) == [0, 2, 2, 0, 0]

    res = dc.mutate(test=2 & strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 2, 2, 0, 0]

    res = dc.mutate(test=strlen("val") & strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 2, 3, 4, 5]


def test_or():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 | 2
    assert str(f) == "or()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 | rnd2
    assert str(f) == "or()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 | rnd2
    assert str(f) == "or()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_or_mutate(dc):
    res = dc.mutate(test=strlen("val") | 2).order_by("num").collect("test")
    assert list(res) == [3, 2, 3, 6, 7]

    res = dc.mutate(test=2 | strlen("val")).order_by("num").collect("test")
    assert list(res) == [3, 2, 3, 6, 7]

    res = dc.mutate(test=strlen("val") | strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 2, 3, 4, 5]


def test_xor():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 ^ 2
    assert str(f) == "xor()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 ^ rnd2
    assert str(f) == "xor()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 ^ rnd2
    assert str(f) == "xor()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_xor_mutate(dc):
    res = dc.mutate(test=strlen("val") ^ 2).order_by("num").collect("test")
    assert list(res) == [3, 0, 1, 6, 7]

    res = dc.mutate(test=2 ^ strlen("val")).order_by("num").collect("test")
    assert list(res) == [3, 0, 1, 6, 7]

    res = dc.mutate(test=strlen("val") ^ strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_rshift():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 >> 2
    assert str(f) == "rshift()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 >> rnd2
    assert str(f) == "rshift()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 >> rnd2
    assert str(f) == "rshift()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_rshift_mutate(dc):
    res = dc.mutate(test=strlen("val") >> 2).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 1, 1]

    res = dc.mutate(test=2 >> strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 0, 0, 0, 0]

    res = dc.mutate(test=strlen("val") >> strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_lshift():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 << 2
    assert str(f) == "lshift()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = 2 << rnd2
    assert str(f) == "lshift()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 << rnd2
    assert str(f) == "lshift()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_lshift_mutate(dc):
    res = dc.mutate(test=strlen("val") << 2).order_by("num").collect("test")
    assert list(res) == [4, 8, 12, 16, 20]

    res = dc.mutate(test=2 << strlen("val")).order_by("num").collect("test")
    assert list(res) == [4, 8, 16, 32, 64]

    res = dc.mutate(test=strlen("val") << strlen("val")).order_by("num").collect("test")
    assert list(res) == [2, 8, 24, 64, 160]


def test_lt():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 < rnd2
    assert str(f) == "lt()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_lt_mutate(dc):
    res = dc.mutate(test=strlen("val") < 3).order_by("num").collect("test")
    assert list(res) == [1, 1, 0, 0, 0]

    res = dc.mutate(test=strlen("val") > 3).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 1, 1]

    res = dc.mutate(test=strlen("val") < strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_le():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 <= rnd2
    assert str(f) == "le()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_le_mutate(dc):
    res = dc.mutate(test=strlen("val") <= 3).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 0, 0]

    res = dc.mutate(test=strlen("val") >= 3).order_by("num").collect("test")
    assert list(res) == [0, 0, 1, 1, 1]

    res = dc.mutate(test=strlen("val") <= strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 1, 1]


def test_eq():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 == rnd2
    assert str(f) == "eq()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_eq_mutate(dc):
    res = dc.mutate(test=strlen("val") == 2).order_by("num").collect("test")
    assert list(res) == [0, 1, 0, 0, 0]

    res = dc.mutate(test=strlen("val") == 4).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 1, 0]

    res = dc.mutate(test=strlen("val") == strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 1, 1]


def test_ne():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 != rnd2
    assert str(f) == "ne()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_ne_mutate(dc):
    res = dc.mutate(test=strlen("val") != 2).order_by("num").collect("test")
    assert list(res) == [1, 0, 1, 1, 1]

    res = dc.mutate(test=strlen("val") != 4).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 0, 1]

    res = dc.mutate(test=strlen("val") != strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_gt():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 > rnd2
    assert str(f) == "gt()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_gt_mutate(dc):
    res = dc.mutate(test=strlen("val") > 2).order_by("num").collect("test")
    assert list(res) == [0, 0, 1, 1, 1]

    res = dc.mutate(test=strlen("val") < 4).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 0, 0]

    res = dc.mutate(test=strlen("val") > strlen("val")).order_by("num").collect("test")
    assert list(res) == [0, 0, 0, 0, 0]


def test_ge():
    rnd1, rnd2 = rand(), rand()

    f = rnd1 >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd1]
    assert f.args == []

    f = rnd2 <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd2]
    assert f.args == []

    f = rnd1 >= rnd2
    assert str(f) == "ge()"
    assert f.cols == [rnd1, rnd2]
    assert f.args == []


def test_ge_mutate(dc):
    res = dc.mutate(test=strlen("val") >= 2).order_by("num").collect("test")
    assert list(res) == [0, 1, 1, 1, 1]

    res = dc.mutate(test=strlen("val") <= 4).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 1, 0]

    res = dc.mutate(test=strlen("val") >= strlen("val")).order_by("num").collect("test")
    assert list(res) == [1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "value,inthash",
    [
        [0, 4761183170873013810],
        [1, 10577349846663553072 - (1 << 64)],
        [5, 15228578409069794350 - (1 << 64)],
        [123456, 13379111408315310133 - (1 << 64)],
    ],
)
def test_sqlite_int_hash_64(value, inthash):
    assert sqlite_int_hash_64(value) == inthash


def test_int_hash_64_mutate(dc):
    res = dc.mutate(test=int_hash_64(strlen("val"))).order_by("num").collect("test")
    assert [x & (2**64 - 1) for x in res] == [
        10577349846663553072,
        18198135717204167749,
        9624464864560415994,
        7766709361750702608,
        15228578409069794350,
    ]


@pytest.mark.parametrize(
    "value1,value2,distance",
    [
        [0, 0, 0],
        [0, 1, 1],
        [2, 3, 1],
        [2, 4, 2],
        [0, 2**64 - 1, 64],
        [-(2**63), 2**63 - 1, 64],
        [-(2**63), 2**63, 0],
    ],
)
def test_sqlite_bit_hamming_distance(value1, value2, distance):
    assert sqlite_bit_hamming_distance(value1, value2) == distance


def test_bit_hamming_distance_mutate(dc):
    res = (
        dc.mutate(test=bit_hamming_distance(strlen("val"), 5))
        .order_by("num")
        .collect("test")
    )
    assert list(res) == [1, 3, 2, 1, 0]


@pytest.mark.parametrize(
    "value1,value2,distance",
    [
        ["", "", 0],
        ["", "a", 1],
        ["foo", "foo", 0],
        ["foo", "bar", 3],
        ["foo", "foobar", 3],
        ["karolin", "kathrin", 3],
    ],
)
def test_sqlite_byte_hamming_distance(value1, value2, distance):
    assert sqlite_byte_hamming_distance(value1, value2) == distance


def test_byte_hamming_distance_mutate(dc):
    res = (
        dc.mutate(test=byte_hamming_distance("val", literal("xxx")))
        .order_by("num")
        .collect("test")
    )
    assert list(res) == [2, 1, 0, 1, 2]
