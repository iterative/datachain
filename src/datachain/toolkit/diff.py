from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

from datachain.lib.diff import CompareStatus, get_status_col_name
from datachain.lib.diff import compare as chain_compare
from datachain.query.schema import Column

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


C = Column


def compare(
    left: "DataChain",
    right: "DataChain",
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]] = None,
    compare: Optional[Union[str, Sequence[str]]] = None,
    right_compare: Optional[Union[str, Sequence[str]]] = None,
    added: bool = True,
    deleted: bool = True,
    modified: bool = True,
    same: bool = False,
) -> dict[str, "DataChain"]:
    """Comparing two chains and returning multiple chains, one for each of `added`,
    `deleted`, `modified` and `same` status. Result is returned in form of
    dictionary where each item represents one of the statuses and key values
    are `A`, `D`, `M`, `S` corresponding. Note that status column is not in the
    resulting chains.

    Parameters:
        left: Chain to calculate diff on.
        right: Chain to calculate diff from.
        on: Column or list of columns to match on. If both chains have the
            same columns then this column is enough for the match. Otherwise,
            `right_on` parameter has to specify the columns for the other chain.
            This value is used to find corresponding row in other dataset. If not
            found there, row is considered as added (or removed if vice versa), and
            if found then row can be either modified or same.
        right_on: Optional column or list of columns
            for the `other` to match.
        compare: Column or list of columns to compare on. If both chains have
            the same columns then this column is enough for the compare. Otherwise,
            `right_compare` parameter has to specify the columns for the other
            chain. This value is used to see if row is modified or same. If
            not set, all columns will be used for comparison
        right_compare: Optional column or list of columns
                for the `other` to compare to.
        added (bool): Whether to return chain containing only added rows.
        deleted (bool): Whether to return chain containing only deleted rows.
        modified (bool): Whether to return chain containing only modified rows.
        same (bool): Whether to return chain containing only same rows.

    Example:
        ```py
        chains = compare(
            persons,
            new_persons,
            on=["id"],
            right_on=["other_id"],
            compare=["name"],
            added=True,
            deleted=True,
            modified=True,
            same=True,
        )
        ```
    """
    status_col = get_status_col_name()

    res = chain_compare(
        left,
        right,
        on,
        right_on=right_on,
        compare=compare,
        right_compare=right_compare,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        status_col=status_col,
    )

    chains = {}

    def filter_by_status(compare_status) -> "DataChain":
        return res.filter(C(status_col) == compare_status).select_except(status_col)

    if added:
        chains[CompareStatus.ADDED.value] = filter_by_status(CompareStatus.ADDED)
    if deleted:
        chains[CompareStatus.DELETED.value] = filter_by_status(CompareStatus.DELETED)
    if modified:
        chains[CompareStatus.MODIFIED.value] = filter_by_status(CompareStatus.MODIFIED)
    if same:
        chains[CompareStatus.SAME.value] = filter_by_status(CompareStatus.SAME)

    return chains
