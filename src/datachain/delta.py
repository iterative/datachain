from typing import TYPE_CHECKING, Optional

from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


def _append_steps(dc: "DataChain", other: "DataChain"):
    """Returns cloned chain with appended steps from other chain.
    Steps are all those modification methods applied like filters, mappers etc.
    """
    dc = dc.clone()
    dc._query.steps += other._query.steps.copy()
    dc.signals_schema = dc.signals_schema.append(other.signals_schema)
    return dc


def delta_update(dc: "DataChain", name: str) -> Optional["DataChain"]:
    """
    Creates new chain that consists of the last version of current delta dataset
    plus diff from the source with all needed modifications.
    This way we don't need to re-calculate the whole chain from the source again(
    apply all the DataChain methods like filters, mappers, generators etc.)
    but just the diff part which is very important for performance.
    """
    from datachain.lib.dc import DataChain

    file_signal = dc.signals_schema.get_file_signal()
    if not file_signal:
        raise ValueError("Datasets without file signals cannot have delta updates")
    try:
        latest_version = dc.session.catalog.get_dataset(name).latest_version
    except DatasetNotFoundError:
        # first creation of delta update dataset
        return None

    source_ds_name = dc._query.starting_step.dataset_name
    source_ds_version = dc._query.starting_step.dataset_version

    diff = DataChain.from_dataset(source_ds_name, version=source_ds_version).diff(
        DataChain.from_dataset(name, version=latest_version),
        on=file_signal,
    )

    # We append all the steps from the original chain to diff, e.g filters, mappers.
    diff = _append_steps(diff, dc)

    # merging diff and the latest version of dataset
    return (
        DataChain.from_dataset(name, latest_version)
        .diff(diff, added=True, modified=False)
        .union(diff)
    )


def delta_update_alternative(dc: "DataChain", name: str) -> Optional["DataChain"]:
    from datachain.lib.dc import DataChain

    catalog = dc.session.catalog
    try:
        latest_version = catalog.get_dataset(name).latest_version
    except DatasetNotFoundError:
        # first creation of delta update dataset
        return None

    dependencies = catalog.get_dataset_dependencies(name, latest_version)
    if len(dependencies) > 1:
        raise Exception("Cannot do delta with dataset that has multiple dependencies")

    dep = dependencies[0]
    if not dep:
        # starting dataset (e.g listing) was removed so we are backing off to normal
        # dataset creation, as it was created first time
        return None

    source_ds_name = dep.name
    source_ds_version = int(dep.version)
    source_ds_latest_version = catalog.get_dataset(source_ds_name).latest_version

    source_dc = DataChain.from_dataset(source_ds_name, source_ds_version)
    source_dc_latest = DataChain.from_dataset(source_ds_name, source_ds_latest_version)
    file_signal = source_dc.signals_schema.get_file_signal()
    if not file_signal:
        raise ValueError("Datasets without file signals cannot have delta updates")

    diff = source_dc_latest.diff(source_dc, on=file_signal)
    # We append all the steps from the original chain to diff, e.g filters, mappers.
    diff = _append_steps(diff, dc)

    # merging diff and the latest version of dataset
    return (
        DataChain.from_dataset(name, latest_version)
        .diff(diff, added=True, modified=False)
        .union(diff)
    )
