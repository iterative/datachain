import numpy as np

from datachain.sql.sqlite.types import adapt_np_array


def test_adapt_np_array():
    arr = np.array([0.5, 0.6], dtype=float)
    assert adapt_np_array(arr) == "[0.5,0.6]"


def test_adapt_nested_np_array():
    arr = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float)
    assert adapt_np_array(arr) == "[[0.5,0.6],[0.7,0.8]]"


def test_adapt_np_array_object_type():
    arr = np.array([0.5, 0.6], dtype=np.dtypes.ObjectDType)
    assert adapt_np_array(arr) == "[0.5, 0.6]"
