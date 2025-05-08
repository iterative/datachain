import pytest

import datachain as dc
from datachain.lib.dc import DatasetPrepareError
from datachain.lib.udf import Mapper


def test_udf(monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)

    class MyMapper(Mapper):
        DEFAULT_VALUE = 84
        BOOTSTRAP_VALUE = 1452
        TEARDOWN_VALUE = 98763

        def __init__(self):
            super().__init__()
            self.value = MyMapper.DEFAULT_VALUE
            self._had_teardown = False

        def process(self, key) -> int:
            return self.value

        def setup(self):
            self.value = MyMapper.BOOTSTRAP_VALUE

        def teardown(self):
            self.value = MyMapper.TEARDOWN_VALUE

    vals = ["a", "b", "c", "d", "e", "f"]
    chain = dc.read_values(key=vals)

    udf = MyMapper()
    res = list(chain.map(res=udf).collect("res"))

    assert res == [MyMapper.BOOTSTRAP_VALUE] * len(vals)
    assert udf.value == MyMapper.TEARDOWN_VALUE


def test_no_bootstrap_for_callable():
    class MyMapper:
        def __init__(self):
            self._had_bootstrap = False
            self._had_teardown = False

        def __call__(self, key):
            return None

        def bootstrap(self):
            self._had_bootstrap = True

        def teardown(self):
            self._had_teardown = True

    udf = MyMapper()

    chain = dc.read_values(key=["a", "b", "c"])
    list(chain.map(res=udf).collect())

    assert udf._had_bootstrap is False
    assert udf._had_teardown is False


def test_bootstrap_in_chain():
    base = 1278
    prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    res = list(
        dc.read_values(val=prime)
        .setup(init_val=lambda: base)
        .map(x=lambda val, init_val: val + init_val, output=int)
        .order_by("x")
        .collect("x")
    )

    assert res == [base + val for val in prime]


def test_vars_duplication_error():
    with pytest.raises(DatasetPrepareError):
        (
            dc.read_values(val=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
            .setup(init_val=lambda: 11, connection=lambda: 123)
            .setup(init_val=lambda: 599)
        )
