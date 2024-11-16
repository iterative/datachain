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
        [[1, 1], [[2, 3, 5], [1, 4, 6, 7, 8, 9, 10]]],
        [[4, 1], [[2, 3, 4, 5, 7, 8, 9], [1, 6, 10]]],
        [[0.7, 0.2, 0.1], [[2, 3, 4, 5, 8, 9], [1, 6, 7], [10]]],
    ],
)
def test_train_test_split_random(pseudo_random_ds, weights, expected):
    res = train_test_split(pseudo_random_ds, weights)
    assert len(res) == len(expected)

    for i, dc in enumerate(res):
        assert list(dc.collect("sys.id")) == expected[i]
