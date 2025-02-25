from typing import TYPE_CHECKING, Optional

from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


def delta_update(dc: "DataChain", name: str) -> Optional["DataChain"]:
    from datachain.lib.dc import DataChain

    file_signal = dc.signals_schema.get_file_signal()
    if not file_signal:
        raise ValueError("Datasets without file signals cannot have delta updates")
    try:
        latest_version = dc.session.catalog.get_dataset(name).latest_version
    except DatasetNotFoundError:
        return None

    source_ds_name = dc._query.starting_step.dataset_name
    source_ds_version = dc._query.starting_step.dataset_version
    diff = DataChain.from_dataset(source_ds_name, version=source_ds_version).diff(
        DataChain.from_dataset(name, version=latest_version), on=file_signal
    )
    # we append all the steps from original chain to diff,
    # e.g filters, mappers, generators etc.
    diff._query.steps += dc._query.steps

    # merging diff and latest version of our dataset chains
    return diff.union(DataChain.from_dataset(name, latest_version))
