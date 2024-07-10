import pytest

from datachain.query import metrics


@pytest.mark.parametrize("value", [42, None, False, 3.14, "bar"])
def test_query_metrics(value, mocker):
    mocker.patch.dict("datachain.query.metrics.metrics", {}, clear=True)

    metrics.set("foo", value)
    assert metrics.get("foo") == value
    assert metrics.metrics == {"foo": value}


@pytest.mark.parametrize("key", [42, None, False, 3.14, [], {}, test_query_metrics])
def test_query_metrics_bad_key(key):
    with pytest.raises(TypeError, match="Key must be a string"):
        metrics.set(key, 12)


def test_query_metrics_empty_key():
    with pytest.raises(ValueError, match="Key must not be empty"):
        metrics.set("", 12)


@pytest.mark.parametrize("value", [[], {}, test_query_metrics])
def test_query_metrics_bad_value(value):
    with pytest.raises(TypeError, match="Value must be a string, int, float or bool"):
        metrics.set("foo", value)
