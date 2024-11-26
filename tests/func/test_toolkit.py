import pytest

from datachain.toolkit import train_test_split


@pytest.mark.parametrize(
    "weights,expected",
    [
        [[1, 1], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]],
        [[4, 1], [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]],
        [[0.7, 0.2, 0.1], [[1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]],
    ],
)
def test_train_test_split_not_random(not_random_ds, weights, expected):
    res = train_test_split(not_random_ds, weights)
    assert len(res) == len(expected)

    for i, dc in enumerate(res):
        assert list(dc.collect("sys.id")) == expected[i]


@pytest.mark.parametrize(
    "weights,expected",
    [
        [[1, 1], [[1, 5, 6, 7, 8], [2, 3, 4, 9, 10]]],
        [[4, 1], [[1, 3, 5, 6, 7, 8, 9], [2, 4, 10]]],
        [[0.7, 0.2, 0.1], [[1, 3, 5, 6, 7, 8, 9], [2, 4], [10]]],
    ],
)
def test_train_test_split_random(pseudo_random_ds, weights, expected):
    res = train_test_split(pseudo_random_ds, weights)
    assert len(res) == len(expected)

    for i, dc in enumerate(res):
        assert list(dc.collect("sys.id")) == expected[i]


def test_train_test_split_errors(not_random_ds):
    with pytest.raises(ValueError, match="Weights should have at least two elements"):
        train_test_split(not_random_ds, [0.5])
    with pytest.raises(ValueError, match="Weights should be non-negative"):
        train_test_split(not_random_ds, [-1, 1])
