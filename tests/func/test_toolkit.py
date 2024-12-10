import pytest

from datachain.toolkit import train_test_split


@pytest.mark.parametrize(
    "seed,weights,expected",
    [
        [None, [1, 1], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]],
        [None, [4, 1], [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]],
        [None, [0.7, 0.2, 0.1], [[1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]],
        [0, [1, 1], [[5, 6, 8], [1, 2, 3, 4, 7, 9, 10]]],
        [1, [1, 1], [[1, 2, 3, 4, 6, 7, 8, 10], [5, 9]]],
        [1234567890, [1, 1], [[2, 3, 5], [1, 4, 6, 7, 8, 9, 10]]],
    ],
)
def test_train_test_split_not_random(not_random_ds, seed, weights, expected):
    res = train_test_split(not_random_ds, weights, seed=seed)
    assert len(res) == len(expected)

    for i, dc in enumerate(res):
        assert list(dc.collect("sys.id")) == expected[i]


@pytest.mark.parametrize(
    "seed,weights,expected",
    [
        [None, [1, 1], [[4, 5, 7, 8], [1, 2, 3, 6, 9, 10]]],
        [None, [4, 1], [[1, 2, 4, 5, 7, 8, 10], [3, 6, 9]]],
        [None, [0.7, 0.2, 0.1], [[1, 2, 4, 5, 7, 8, 10], [3, 6, 9], []]],
        [0, [1, 1], [[4, 7, 8, 10], [1, 2, 3, 5, 6, 9]]],
        [1, [1, 1], [[3, 4, 6, 7, 10], [1, 2, 5, 8, 9]]],
        [1234567890, [1, 1], [[1, 2, 3, 4, 5, 6, 8], [7, 9, 10]]],
        [0, [4, 1], [[1, 3, 4, 5, 7, 8, 10], [2, 6, 9]]],
        [1, [4, 1], [[2, 3, 4, 5, 6, 7, 8, 9, 10], [1]]],
        [1234567890, [4, 1], [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], []]],
    ],
)
def test_train_test_split_random(pseudo_random_ds, seed, weights, expected):
    res = train_test_split(pseudo_random_ds, weights, seed=seed)
    assert len(res) == len(expected)

    for i, dc in enumerate(res):
        assert list(dc.collect("sys.id")) == expected[i]


def test_train_test_split_errors(not_random_ds):
    with pytest.raises(ValueError, match="Weights should have at least two elements"):
        train_test_split(not_random_ds, [0.5])
    with pytest.raises(ValueError, match="Weights should be non-negative"):
        train_test_split(not_random_ds, [-1, 1])
