from typing import TYPE_CHECKING, Optional

from datachain.error import DatasetNotFoundError

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


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
        DataChain.from_dataset(name, version=latest_version), on=file_signal
    )
    # we append all the steps from the original chain to diff,
    # e.g filters, mappers, generators etc. With this we make sure we add all
    # needed modifications to diff part as well
    diff._query.steps += dc._query.steps

    # merging diff and the latest version of our dataset
    return diff.union(DataChain.from_dataset(name, latest_version))
