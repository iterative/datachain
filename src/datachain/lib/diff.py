import random
import string
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

import sqlalchemy as sa

from datachain.lib.signal_schema import SignalSchema
from datachain.query.schema import Column
from datachain.sql.types import String

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


C = Column


def compare(  # noqa: PLR0912, PLR0915, C901
    left: "DataChain",
    right: "DataChain",
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]] = None,
    compare: Optional[Union[str, Sequence[str]]] = None,
    right_compare: Optional[Union[str, Sequence[str]]] = None,
    added: bool = True,
    deleted: bool = True,
    modified: bool = True,
    same: bool = True,
    status_col: Optional[str] = None,
) -> "DataChain":
    """Comparing two chains by identifying rows that are added, deleted, modified
    or same"""
    dialect = left._query.dialect

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

    if not any([added, deleted, modified, same]):
        raise ValueError(
            "At least one of added, deleted, modified, same flags must be set"
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
    cols = left.signals_schema.clone_without_sys_signals().db_signals()
    right_cols = right.signals_schema.clone_without_sys_signals().db_signals()

    on = left.signals_schema.resolve(*on).db_signals()  # type: ignore[assignment]
    right_on = right.signals_schema.resolve(*right_on).db_signals()  # type: ignore[assignment]
    if compare:
        right_compare = right_compare or compare
        compare = left.signals_schema.resolve(*compare).db_signals()  # type: ignore[assignment]
        right_compare = right.signals_schema.resolve(*right_compare).db_signals()  # type: ignore[assignment]
    elif not compare and len(cols) != len(right_cols):
        # here we will mark all rows that are not added or deleted as modified since
        # there was no explicit list of compare columns provided (meaning we need
        # to check all columns to determine if row is modified or same), but
        # the number of columns on left and right is not the same (one of the chains
        # have additional column)
        compare = None
        right_compare = None
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
    if modified and compare:
        modified_cond = sa.or_(
            *[
                C(c) != C(f"{_rprefix(c, rc)}{rc}")
                for c, rc in zip(compare, right_compare)  # type: ignore[arg-type]
            ]
        )
        diff_cond.append((modified_cond, "M"))
    if same and compare:
        same_cond = sa.and_(
            *[
                C(c) == C(f"{_rprefix(c, rc)}{rc}")
                for c, rc in zip(compare, right_compare)  # type: ignore[arg-type]
            ]
        )
        diff_cond.append((same_cond, "S"))

    diff = sa.case(*diff_cond, else_=None if compare else "M").label(status_col)
    diff.type = String()

    left_right_merge = left.merge(
        right, on=on, right_on=right_on, inner=False, rname=rname
    )
    left_right_merge_select = left_right_merge._query.select(
        *(
            [C(c) for c in left_right_merge.signals_schema.db_signals("sys")]
            + [C(c) for c in on]
            + [C(c) for c in cols if c not in on]
            + [diff]
        )
    )

    diff_col = sa.literal("D").label(status_col)
    diff_col.type = String()

    right_left_merge = right.merge(
        left, on=right_on, right_on=on, inner=False, rname=rname
    ).filter(
        sa.and_(
            *[C(f"{_rprefix(c, rc)}{c}") == None for c, rc in zip(on, right_on)]  # noqa: E711
        )
    )

    def _default_val(chain: "DataChain", col: str):
        col_type = chain._query.column_types[col]  # type: ignore[index]
        val = sa.literal(col_type.default_value(dialect)).label(col)
        val.type = col_type()
        return val

    right_left_merge_select = right_left_merge._query.select(
        *(
            [C(c) for c in right_left_merge.signals_schema.db_signals("sys")]
            + [
                C(c) if c == rc else _default_val(left, c)
                for c, rc in zip(on, right_on)
            ]
            + [
                C(c) if c in right_cols else _default_val(left, c)  # type: ignore[arg-type]
                for c in cols
                if c not in on
            ]
            + [diff_col]
        )
    )

    if not deleted:
        res = left_right_merge_select
    elif deleted and not any([added, modified, same]):
        res = right_left_merge_select
    else:
        res = left_right_merge_select.union(right_left_merge_select)

    res = res.filter(C(status_col) != None)  # noqa: E711

    schema = left.signals_schema
    if need_status_col:
        res = res.select()
        schema = SignalSchema({status_col: str}) | schema
    else:
        res = res.select_except(C(status_col))

    return left._evolve(query=res, signal_schema=schema)
