from typing import get_args, get_origin

import pytest

from datachain.lib.dc import DataChain
from datachain.lib.feature_utils import FeatureToTupleError, features_to_tuples
from datachain.query.schema import Column


def test_basic():
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    typ, _partition_by, vals = features_to_tuples(fib=fib, odds=values)

    assert get_origin(typ) is tuple
    assert get_args(typ) == (int, str)
    assert len(vals) == len(fib)
    assert type(vals[0]) is tuple
    assert len(vals[0]) == 2
    assert vals[0] == (1, "odd")
    assert vals[-1] == (fib[-1], values[-1])


def test_e2e(catalog):
    fib = [1, 1, 2, 3, 5, 8]
    values = ["odd" if num % 2 else "even" for num in fib]

    dc = DataChain.from_features(fib=fib, odds=values)

    vals = list(dc.iterate())
    lst1 = [item[0] for item in vals]
    lst2 = [item[1] for item in vals]

    assert lst1 == fib
    assert lst2 == values


def test_single_value():
    fib = [1, 1, 2, 3, 5, 8]

    typ, _partition_by, vals = features_to_tuples(fib=fib)

    assert typ is int
    assert vals == fib


def test_single_e2e(catalog):
    fib = [1, 1, 2, 3, 5, 8]

    dc = DataChain.from_features(fib=fib)

    vals = list(dc.iterate())
    flattened = [item for sublist in vals for item in sublist]

    assert flattened == fib


def test_not_array_value_error():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value=True)


def test_empty_value_list_error():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value=[])


def test_features_length_missmatch():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value1=[1, 2, 3], value2=[1, 2, 3, 4, 5])


def test_unknown_output_type():
    with pytest.raises(FeatureToTupleError):

        class UnknownFrType:
            def __init__(self, val):
                self.val = val

        DataChain.from_features(value1=[UnknownFrType(1), UnknownFrType(23)])


def test_output_type_missmatch():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value1=[1, 2, 3], output={"res": str})


def test_output_length_missmatch():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value1=[1, 2, 3], output={"out1": int, "out2": int})


def test_output_spec_wrong_type():
    with pytest.raises(FeatureToTupleError):
        DataChain.from_features(value1=[1, 2, 3], output=123)


def test_resolve_column():
    signal = Column("hello.world.again")
    assert signal.name == "hello__world__again"


def test_resolve_column_attr():
    signal = Column.hello.world.again
    assert signal.name == "hello__world__again"
