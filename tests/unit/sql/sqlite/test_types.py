import numpy as np
import pytest
import ujson as json

from datachain.sql.sqlite.types import adapt_np_array


@pytest.mark.parametrize(
    "dtype,arr,expected",
    (
        (float, [], "[]"),
        (float, [0.5, 0.6], "[0.5,0.6]"),
        (float, [[0.5, 0.6], [0.7, 0.8]], "[[0.5,0.6],[0.7,0.8]]"),
        (np.dtypes.ObjectDType, [], "[]"),
        (np.dtypes.ObjectDType, [0.5, 0.6], "[0.5,0.6]"),
        (np.dtypes.ObjectDType, [[0.5, 0.6], [0.7, 0.8]], "[[0.5,0.6],[0.7,0.8]]"),
    ),
)
def test_adapt_np_array(dtype, arr, expected):
    assert adapt_np_array(np.array(arr, dtype=dtype)) == expected


def test_adapt_np_array_nan_inf():
    arr_with_nan = np.array([1.0, np.nan, 3.0])
    result = adapt_np_array(arr_with_nan)
    assert result == "[1.0,NaN,3.0]"

    arr_with_inf = np.array([1.0, np.inf, -np.inf])
    result = adapt_np_array(arr_with_inf)
    assert result == "[1.0,Infinity,-Infinity]"

    arr_2d = np.array([[np.nan, 1.0], [2.0, np.inf]])
    result = adapt_np_array(arr_2d)
    assert result == "[[NaN,1.0],[2.0,Infinity]]"

    parsed = json.loads(result)
    assert np.isnan(parsed[0][0])
    assert parsed[0][1] == 1.0
    assert parsed[1][0] == 2.0
    assert np.isinf(parsed[1][1])
