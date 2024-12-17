import random
import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

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
    unchanged: bool = False,
) -> dict[str, "DataChain"]:
    from datachain.lib.diff import compare as chain_compare

    status_col = "diff_" + "".join(
        random.choice(string.ascii_letters)  # noqa: S311
        for _ in range(10)
    )

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
        unchanged=unchanged,
        status_col=status_col,
    )

    chains = {}
    if added:
        chains["A"] = res.filter(C(status_col) == "A").select_except(status_col)
    if deleted:
        chains["D"] = res.filter(C(status_col) == "D").select_except(status_col)
    if modified:
        chains["M"] = res.filter(C(status_col) == "M").select_except(status_col)
    if unchanged:
        chains["U"] = res.filter(C(status_col) == "U").select_except(status_col)

    return chains
