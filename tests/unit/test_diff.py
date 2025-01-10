import pytest

from datachain.diff import CompareStatus, compare_and_split
from datachain.lib.dc import DataChain


@pytest.mark.parametrize("added", (True, False))
@pytest.mark.parametrize("deleted", (True, False))
@pytest.mark.parametrize("modified", (True, False))
@pytest.mark.parametrize("same", (True, False))
def test_compare(test_session, added, deleted, modified, same):
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

    if not any([added, deleted, modified, same]):
        with pytest.raises(ValueError) as exc_info:
            compare_and_split(
                ds1,
                ds2,
                added=added,
                deleted=deleted,
                modified=modified,
                same=same,
                on=["id"],
            )
        assert str(exc_info.value) == (
            "At least one of added, deleted, modified, same flags must be set"
        )
        return

    chains = compare_and_split(
        ds1,
        ds2,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        on=["id"],
    )

    collect_fields = ["id", "name"]
    if added:
        assert "diff" not in chains[CompareStatus.ADDED].signals_schema.db_signals()
        assert list(
            chains[CompareStatus.ADDED].order_by("id").collect(*collect_fields)
        ) == [(2, "Doe")]
    if deleted:
        assert "diff" not in chains[CompareStatus.DELETED].signals_schema.db_signals()
        assert list(
            chains[CompareStatus.DELETED].order_by("id").collect(*collect_fields)
        ) == [(3, "Mark")]
    if modified:
        assert "diff" not in chains[CompareStatus.MODIFIED].signals_schema.db_signals()
        assert list(
            chains[CompareStatus.MODIFIED].order_by("id").collect(*collect_fields)
        ) == [(1, "John1")]
    if same:
        assert "diff" not in chains[CompareStatus.SAME].signals_schema.db_signals()
        assert list(
            chains[CompareStatus.SAME].order_by("id").collect(*collect_fields)
        ) == [(4, "Andy")]
