import pytest
from sqlalchemy import Label

from datachain import func
from datachain.lib.signal_schema import SignalSchema


@pytest.fixture
def rnd():
    return func.random.rand()


def test_db_cols(rnd):
    assert rnd._db_cols == []
    assert rnd._db_col_type(SignalSchema({})) is None


def test_label(rnd):
    assert rnd.col_label is None
    assert rnd.label("test2") == "test2"

    f = rnd.label("test")
    assert f.col_label == "test"
    assert f.label("test2") == "test2"


def test_col_name(rnd):
    assert rnd.get_col_name() == "rand"
    assert rnd.label("test").get_col_name() == "test"
    assert rnd.get_col_name("test2") == "test2"


def test_result_type(rnd):
    assert rnd.get_result_type(SignalSchema({})) is int


def test_get_column(rnd):
    col = rnd.get_column(SignalSchema({}))
    assert isinstance(col, Label)
    assert col.name == "rand"


def test_add(rnd):
    f = rnd + 1
    assert str(f) == "add()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 + rnd
    assert str(f) == "add()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_sub(rnd):
    f = rnd - 1
    assert str(f) == "sub()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 - rnd
    assert str(f) == "sub()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_mul(rnd):
    f = rnd * 1
    assert str(f) == "mul()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 * rnd
    assert str(f) == "mul()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_realdiv(rnd):
    f = rnd / 1
    assert str(f) == "div()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 / rnd
    assert str(f) == "div()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_floordiv(rnd):
    f = rnd // 1
    assert str(f) == "floordiv()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 // rnd
    assert str(f) == "floordiv()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_mod(rnd):
    f = rnd % 1
    assert str(f) == "mod()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 % rnd
    assert str(f) == "mod()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_pow(rnd):
    f = rnd**1
    assert str(f) == "pow()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1**rnd
    assert str(f) == "pow()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_lshift(rnd):
    f = rnd << 1
    assert str(f) == "lshift()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 << rnd
    assert str(f) == "lshift()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_rshift(rnd):
    f = rnd >> 1
    assert str(f) == "rshift()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 >> rnd
    assert str(f) == "rshift()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_and(rnd):
    f = rnd & 1
    assert str(f) == "and()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 & rnd
    assert str(f) == "and()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_or(rnd):
    f = rnd | 1
    assert str(f) == "or()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 | rnd
    assert str(f) == "or()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_xor(rnd):
    f = rnd ^ 1
    assert str(f) == "xor()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = 1 ^ rnd
    assert str(f) == "xor()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_lt(rnd):
    f = rnd < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_le(rnd):
    f = rnd <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_eq(rnd):
    f = rnd == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd == 1
    assert str(f) == "eq()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_ne(rnd):
    f = rnd != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd != 1
    assert str(f) == "ne()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_gt(rnd):
    f = rnd > 1
    assert str(f) == "gt()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd < 1
    assert str(f) == "lt()"
    assert f.cols == [rnd]
    assert f.args == [1]


def test_ge(rnd):
    f = rnd >= 1
    assert str(f) == "ge()"
    assert f.cols == [rnd]
    assert f.args == [1]

    f = rnd <= 1
    assert str(f) == "le()"
    assert f.cols == [rnd]
    assert f.args == [1]
