from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

from datachain import read_dataset
from datachain.error import DatasetNotFoundError
from datachain.lib.dc import C

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain


def retry_update(
    dc: "DataChain",
    name: str,
    on: Union[str, Sequence[str]],
    right_on: Optional[Union[str, Sequence[str]]] = None,
    retry_on: Optional[str] = None,
    retry_missing: bool = False,
) -> tuple[Optional["DataChain"], bool]:
    """
    Creates a chain that consists of records from source dataset that need to be
    reprocessed.
    These are records that either:
    1. Have a non-None value in the field specified by retry_on in the result dataset
    2. Exist in the source dataset but are missing in the result dataset
       (if retry_missing=True)

    Parameters:
        dc: The DataChain to filter for records that need reprocessing
        name: Name of the destination dataset
        on: Field(s) in source dataset that uniquely identify records
        right_on: Corresponding field(s) in result dataset if they differ from source
        retry_on: Field in result dataset that indicates an error when not None
        retry_missing: If True, also include records missing from result dataset

    Returns:
        A tuple containing (filtered chain for reprocessing, found records flag)
    """
    catalog = dc.session.catalog
    dc._query.apply_listing_pre_step()

    try:
        latest_version = catalog.get_dataset(name).latest_version
    except DatasetNotFoundError:
        # First creation of result dataset, return all source records
        return dc, True

    # Read the latest version of the result dataset
    result_dataset = read_dataset(name, latest_version)

    # Create a chain containing records to reprocess
    if retry_on is not None:
        # Get records with non-None values in retry_on field
        error_records = result_dataset.filter(C(retry_on) != "")
    else:
        # No error records if retry_on is not provided
        error_records = None

    # Initialize reprocess_chain as None
    reprocess_chain = None

    # Handle error records if they exist
    if error_records is not None:
        # Use merge (inner join) to find source records that match error records
        error_source_records = dc.merge(
            error_records, on=on, right_on=right_on, inner=True
        ).select_except(
            *[col for col in error_records.signals_schema.values if col != "sys"]
        )

        reprocess_chain = error_source_records

    # Handle missing records if retry_missing is True
    if retry_missing:
        # Create a subtract operation to find missing records
        missing_records = dc.subtract(result_dataset, on=on, right_on=right_on)

        if not missing_records.empty:
            if reprocess_chain is not None:
                # Union the error records and missing records
                reprocess_chain = reprocess_chain.union(missing_records)
            else:
                reprocess_chain = missing_records

    # If no records to reprocess, return None
    if reprocess_chain is None:
        return None, False

    # Return the chain with records to reprocess
    return reprocess_chain, True
