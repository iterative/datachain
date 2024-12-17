import pytest

from datachain.lib.dc import DataChain
from datachain.toolkit import compare, train_test_split


@pytest.mark.parametrize(
    "seed,weights,expected",
    [
        [None, [1, 1], [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]],
        [None, [4, 1], [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]],
        [None, [0.7, 0.2, 0.1], [[1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]],
        [0, [1, 1], [[3, 5], [1, 2, 4, 6, 7, 8, 9, 10]]],
        [1, [1, 1], [[1, 3, 4, 6, 9, 10], [2, 5, 7, 8]]],
        [1234567890, [1, 1], [[2, 4, 5, 7, 9], [1, 3, 6, 8, 10]]],
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
        [None, [1, 1], [[1, 5, 6, 7, 8], [2, 3, 4, 9, 10]]],
        [None, [4, 1], [[1, 3, 5, 6, 7, 8, 9], [2, 4, 10]]],
        [None, [0.7, 0.2, 0.1], [[1, 3, 5, 6, 7, 8, 9], [2, 4], [10]]],
        [0, [1, 1], [[2, 5, 9, 10], [1, 3, 4, 6, 7, 8]]],
        [1, [1, 1], [[1, 2, 3, 4, 5, 6, 8], [7, 9, 10]]],
        [1234567890, [1, 1], [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]],
        [0, [4, 1], [[1, 2, 4, 5, 7, 9, 10], [3, 6, 8]]],
        [1, [4, 1], [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10]]],
        [1234567890, [4, 1], [[1, 3, 5, 6, 7, 9, 10], [2, 4, 8]]],
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


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("unchanged", (True, False))
def test_compare(test_session, added, deleted, modified, unchanged):
    ds1 = DataChain.from_values(
        id=[1, 2, 4],
        name=["John1", "Doe", "Andy"],
        session=test_session,
    ).save("ds1")

    ds2 = DataChain.from_values(
        id=[1, 3, 4],
        name=["John", "Mark", "Andy"],
        session=test_session,
    ).save("ds2")

    if not any([added, deleted, modified, unchanged]):
        with pytest.raises(ValueError) as exc_info:
            compare(
                ds1,
                ds2,
                added=added,
                deleted=deleted,
                modified=modified,
                unchanged=unchanged,
                on=["id"],
            )
        assert str(exc_info.value) == (
            "At least one of added, deleted, modified, unchanged flags must be set"
        )
        return

    chains = compare(
        ds1,
        ds2,
        added=added,
        deleted=deleted,
        modified=modified,
        unchanged=unchanged,
        on=["id"],
    )

    collect_fields = ["id", "name"]
    if added:
        assert "diff" not in chains["A"].signals_schema.db_signals()
        assert list(chains["A"].order_by("id").collect(*collect_fields)) == [(2, "Doe")]
    if deleted:
        assert "diff" not in chains["D"].signals_schema.db_signals()
        assert list(chains["D"].order_by("id").collect(*collect_fields)) == [
            (3, "Mark")
        ]
    if modified:
        assert "diff" not in chains["M"].signals_schema.db_signals()
        assert list(chains["M"].order_by("id").collect(*collect_fields)) == [
            (1, "John1")
        ]
    if unchanged:
        assert "diff" not in chains["U"].signals_schema.db_signals()
        assert list(chains["U"].order_by("id").collect(*collect_fields)) == [
            (4, "Andy")
        ]
