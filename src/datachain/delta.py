from collections.abc import Sequence
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import datachain
from datachain.dataset import DatasetDependency
from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from typing_extensions import Concatenate, ParamSpec

    from datachain.lib.dc import DataChain

    P = ParamSpec("P")


T = TypeVar("T", bound="DataChain")


def delta_disabled(
    method: "Callable[Concatenate[T, P], T]",
) -> "Callable[Concatenate[T, P], T]":
    """
    Decorator for disabling DataChain methods (e.g `.agg()` or `.union()`) to
    work with delta updates. It throws `NotImplementedError` if chain on which
    method is called is marked as delta.
    """

    @wraps(method)
    def _inner(self: T, *args: "P.args", **kwargs: "P.kwargs") -> T:
        if self.delta:
            raise NotImplementedError(
                f"Delta update cannot be used with {method.__name__}"
            )
        return method(self, *args, **kwargs)

    return _inner


def _append_steps(dc: "DataChain", other: "DataChain"):
    """Returns cloned chain with appended steps from other chain.
    Steps are all those modification methods applied like filters, mappers etc.
    """
    dc = dc.clone()
    dc._query.steps += other._query.steps.copy()
    dc.signals_schema = other.signals_schema
    return dc


def delta_update(
    dc: "DataChain",
    name: str,
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]] = None,
    compare: Optional[Union[str, Sequence[str]]] = None,
) -> tuple[Optional["DataChain"], Optional[list[DatasetDependency]], bool]:
    """
    Creates new chain that consists of the last version of current delta dataset
    plus diff from the source with all needed modifications.
    This way we don't need to re-calculate the whole chain from the source again(
    apply all the DataChain methods like filters, mappers, generators etc.)
    but just the diff part which is very important for performance.

    Note that currently delta update works only if there is only one direct dependency.
    """
    catalog = dc.session.catalog
    dc._query.apply_listing_pre_step()

    try:
        latest_version = catalog.get_dataset(name).latest_version
    except DatasetNotFoundError:
        # first creation of delta update dataset
        return None, None, True

    dependencies = catalog.get_dataset_dependencies(
        name, latest_version, indirect=False
    )

    dep = dependencies[0]
    if not dep:
        # starting dataset (e.g listing) was removed so we are backing off to normal
        # dataset creation, as it was created first time
        return None, None, True

    source_ds_name = dep.name
    source_ds_version = dep.version
    source_ds_latest_version = catalog.get_dataset(source_ds_name).latest_version
    dependencies = copy(dependencies)
    dependencies = [d for d in dependencies if d is not None]  # filter out removed dep
    dependencies[0].version = source_ds_latest_version  # type: ignore[union-attr]

    source_dc = datachain.read_dataset(source_ds_name, source_ds_version)
    source_dc_latest = datachain.read_dataset(source_ds_name, source_ds_latest_version)

    diff = source_dc_latest.compare(source_dc, on=on, compare=compare, deleted=False)
    # We append all the steps from the original chain to diff, e.g filters, mappers.
    diff = _append_steps(diff, dc)

    # to avoid re-calculating diff multiple times
    diff = diff.persist()

    if diff.empty:
        return None, None, False

    # merging diff and the latest version of dataset
    delta_chain = (
        datachain.read_dataset(name, latest_version)
        .compare(
            diff,
            on=right_on or on,
            added=True,
            modified=False,
            deleted=False,
        )
        .union(diff)
    )

    return delta_chain, dependencies, True  # type: ignore[return-value]
