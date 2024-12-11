import random
import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

import sqlalchemy as sa

from datachain.lib.signal_schema import SignalSchema
from datachain.query.schema import Column

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


C = Column


def compare(  # noqa: PLR0912, C901
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
    status_col: Optional[str] = None,
) -> "DataChain":
    """Comparing two chains by identifying rows that are added, deleted, modified
    or unchanged"""

    rname = "right_"

    def _rprefix(c: str, rc: str) -> str:
        """Returns prefix of right of two companion left - right columns
        from merge. If companion columns have the same name then prefix will
        be present in right column name, otherwise it won't.
        """
        return rname if c == rc else ""

    def _to_list(obj: Union[str, Sequence[str]]) -> list[str]:
        return [obj] if isinstance(obj, str) else list(obj)

    if on is None:
        raise ValueError("'on' must be specified")

    on = _to_list(on)
    if right_on:
        right_on = _to_list(right_on)
        if len(on) != len(right_on):
            raise ValueError("'on' and 'right_on' must be have the same length")

    if compare:
        compare = _to_list(compare)

    if right_compare:
        if not compare:
            raise ValueError("'compare' must be defined if 'right_compare' is defined")

        right_compare = _to_list(right_compare)
        if len(compare) != len(right_compare):
            raise ValueError(
                "'compare' and 'right_compare' must be have the same length"
            )

    if not any([added, deleted, modified, unchanged]):
        raise ValueError(
            "At least one of added, deleted, modified, unchanged flags must be set"
        )

    # we still need status column for internal implementation even if not
    # needed in output
    need_status_col = bool(status_col)
    status_col = status_col or "diff_" + "".join(
        random.choice(string.ascii_letters)  # noqa: S311
        for _ in range(10)
    )

    # calculate on and compare column names
    right_on = right_on or on
    cols = left.signals_schema.db_signals()
    right_cols = right.signals_schema.db_signals()

    on = left.signals_schema.resolve(*on).db_signals()  # type: ignore[assignment]
    right_on = right.signals_schema.resolve(*right_on).db_signals()  # type: ignore[assignment]
    if compare:
        right_compare = right_compare or compare
        compare = left.signals_schema.resolve(*compare).db_signals()  # type: ignore[assignment]
        right_compare = right.signals_schema.resolve(*right_compare).db_signals()  # type: ignore[assignment]
    else:
        compare = [c for c in cols if c in right_cols]  # type: ignore[misc, assignment]
        right_compare = compare

    diff_cond = []

    if added:
        added_cond = sa.and_(
            *[
                C(c) == None  # noqa: E711
                for c in [f"{_rprefix(c, rc)}{rc}" for c, rc in zip(on, right_on)]
            ]
        )
        diff_cond.append((added_cond, "A"))
    if modified:
        modified_cond = sa.or_(
            *[
                C(c) != C(f"{_rprefix(c, rc)}{rc}")
                for c, rc in zip(compare, right_compare)
            ]
        )
        diff_cond.append((modified_cond, "M"))
    if unchanged:
        unchanged_cond = sa.and_(
            *[
                C(c) == C(f"{_rprefix(c, rc)}{rc}")
                for c, rc in zip(compare, right_compare)
            ]
        )
        diff_cond.append((unchanged_cond, "U"))

    diff = sa.case(*diff_cond, else_=None).label(status_col)

    left_right_merge = left.merge(
        right, on=on, right_on=right_on, inner=False, rname=rname
    )._query.select(
        *([C(c) for c in on] + [C(c) for c in cols if c not in on] + [diff])
    )

    right_left_merge = (
        right.merge(left, on=right_on, right_on=on, inner=False, rname=rname)
        ._query.select(
            *(
                [
                    C(c) if c == rc else sa.literal(None).label(c)
                    for c, rc in zip(on, right_on)
                ]
                + [
                    C(c) if c in right_cols else sa.literal(None).label(c)  # type: ignore[arg-type]
                    for c in cols
                    if c not in on
                ]
                + [sa.literal("D").label(status_col)]
            )
        )
        .filter(
            sa.and_(
                *[
                    C(f"{_rprefix(c, rc)}{c}") == None  # noqa: E711
                    for c, rc in zip(on, right_on)
                ]
            )
        )
    )

    if not deleted:
        res = left_right_merge
    elif deleted and not any([added, modified, unchanged]):
        res = right_left_merge
    else:
        res = left_right_merge.union(right_left_merge)

    res = res.filter(C(status_col) != None)  # noqa: E711

    schema = left.signals_schema
    if need_status_col:
        schema = SignalSchema({status_col: str}) | schema

    return left._evolve(query=res, signal_schema=schema)
