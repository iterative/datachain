from collections.abc import Sequence
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import datachain
from datachain.dataset import DatasetDependency
from datachain.error import DatasetNotFoundError
from datachain.project import Project

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


def _get_delta_chain(
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    source_ds_latest_version: str,
    on: Union[str, Sequence[str]],
    compare: Optional[Union[str, Sequence[str]]] = None,
) -> "DataChain":
    """Get delta chain for processing changes between versions."""
    source_dc = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_version,
    )
    source_dc_latest = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_latest_version,
    )

    # Calculate diff between source versions
    return source_dc_latest.diff(source_dc, on=on, compare=compare, deleted=False)


def _get_retry_chain(
    name: str,
    project: Project,
    latest_version: str,
    source_ds_name: str,
    source_ds_project: Project,
    source_ds_version: str,
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]],
    delta_retry: Optional[Union[bool, str]],
    diff_chain: "DataChain",
) -> Optional["DataChain"]:
    """Get retry chain for processing error records and missing records."""
    # Import here to avoid circular import
    from datachain.lib.dc import C

    retry_chain = None

    # Read the latest version of the result dataset for retry logic
    result_dataset = datachain.read_dataset(
        name,
        namespace=project.namespace.name,
        project=project.name,
        version=latest_version,
    )
    source_dc = datachain.read_dataset(
        source_ds_name,
        namespace=source_ds_project.namespace.name,
        project=source_ds_project.name,
        version=source_ds_version,
    )

    # Handle error records if delta_retry is a string (column name)
    if isinstance(delta_retry, str):
        error_records = result_dataset.filter(C(delta_retry) != "")
        error_source_records = source_dc.merge(
            error_records, on=on, right_on=right_on, inner=True
        ).select(*list(source_dc.signals_schema.values))
        retry_chain = error_source_records

    # Handle missing records if delta_retry is True
    elif delta_retry is True:
        missing_records = source_dc.subtract(result_dataset, on=on, right_on=right_on)
        retry_chain = missing_records

    # Subtract also diff chain since some items might be picked
    # up by `delta=True` itself (e.g. records got modified AND are missing in the
    # result dataset atm)
    return retry_chain.subtract(diff_chain, on=on) if retry_chain else None


def _get_source_info(
    name: str,
    project: Project,
    latest_version: str,
    catalog,
) -> tuple[
    Optional[str],
    Optional[Project],
    Optional[str],
    Optional[str],
    Optional[list[DatasetDependency]],
]:
    """Get source dataset information and dependencies.

    Returns:
        Tuple of (source_name, source_version, source_latest_version, dependencies)
        Returns (None, None, None, None) if source dataset was removed.
    """
    dependencies = catalog.get_dataset_dependencies(
        name, latest_version, project=project, indirect=False
    )

    dep = dependencies[0]
    if not dep:
        # Starting dataset was removed, back off to normal dataset creation
        return None, None, None, None, None

    source_ds_project = catalog.metastore.get_project(dep.project, dep.namespace)
    source_ds_name = dep.name
    source_ds_version = dep.version
    source_ds_latest_version = catalog.get_dataset(
        source_ds_name, project=source_ds_project
    ).latest_version

    return (
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        dependencies,
    )


def delta_retry_update(
    dc: "DataChain",
    namespace_name: str,
    project_name: str,
    name: str,
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]] = None,
    compare: Optional[Union[str, Sequence[str]]] = None,
    delta_retry: Optional[Union[bool, str]] = None,
) -> tuple[Optional["DataChain"], Optional[list[DatasetDependency]], bool]:
    """
    Creates new chain that consists of the last version of current delta dataset
    plus diff from the source with all needed modifications.
    This way we don't need to re-calculate the whole chain from the source again
    (apply all the DataChain methods like filters, mappers, generators etc.)
    but just the diff part which is very important for performance.

    Note that currently delta update works only if there is only one direct
    dependency.

    Additionally supports retry functionality to filter records that either:
    1. Have a non-None value in the field specified by delta_retry (when it's a string)
    2. Exist in the source dataset but are missing in the result dataset
       (when delta_retry=True)

    Parameters:
        dc: The DataChain to filter for records that need reprocessing
        name: Name of the destination dataset
        on: Field(s) in source dataset that uniquely identify records
        right_on: Corresponding field(s) in result dataset if they differ from
                  source
        compare: Field(s) used to check if the same row has been modified
        delta_retry: If string, field in result dataset that indicates an error
                    when not None. If True, include records missing from result dataset.
                    If False/None, no retry functionality.

    Returns:
        A tuple containing (filtered chain for delta/retry processing,
                          dependencies, found records flag)
    """

    catalog = dc.session.catalog
    project = catalog.metastore.get_project(project_name, namespace_name)
    dc._query.apply_listing_pre_step()

    # Check if dataset exists
    try:
        dataset = catalog.get_dataset(name, project=project)
        latest_version = dataset.latest_version
    except DatasetNotFoundError:
        # First creation of result dataset
        return None, None, True

    # Initialize variables
    diff_chain = None
    dependencies = None
    retry_chain = None
    processing_chain = None

    (
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        dependencies,
    ) = _get_source_info(name, project, latest_version, catalog)

    # If source_ds_name is None, starting dataset was removed
    if source_ds_name is None:
        return None, None, True

    assert source_ds_project
    assert source_ds_version
    assert source_ds_latest_version

    diff_chain = _get_delta_chain(
        source_ds_name,
        source_ds_project,
        source_ds_version,
        source_ds_latest_version,
        on,
        compare,
    )

    # Filter out removed dep
    if dependencies:
        dependencies = copy(dependencies)
        dependencies = [d for d in dependencies if d is not None]
        # Update to latest version
        dependencies[0].version = source_ds_latest_version  # type: ignore[union-attr]

    # Handle retry functionality if enabled
    if delta_retry:
        retry_chain = _get_retry_chain(
            name,
            project,
            latest_version,
            source_ds_name,
            source_ds_project,
            source_ds_version,
            on,
            right_on,
            delta_retry,
            diff_chain,
        )

    # Combine delta and retry chains
    if retry_chain is not None:
        processing_chain = diff_chain.union(retry_chain)
    else:
        processing_chain = diff_chain

    # Apply all the steps from the original chain to processing_chain
    processing_chain = _append_steps(processing_chain, dc).persist()

    # Check if chain becomes empty after applying steps
    if processing_chain is None or (processing_chain and processing_chain.empty):
        return None, None, False

    latest_dataset = datachain.read_dataset(
        name,
        namespace=project.namespace.name,
        project=project.name,
        version=latest_version,
    )
    compared_chain = latest_dataset.diff(
        processing_chain,
        on=right_on or on,
        added=True,
        modified=False,
        deleted=False,
    )
    result_chain = compared_chain.union(processing_chain)
    return result_chain, dependencies, True
