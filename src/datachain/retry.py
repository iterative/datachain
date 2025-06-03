from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

from datachain import C, read_dataset

if TYPE_CHECKING:
    from datachain.lib.data_model import DataValue
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
    except Exception:  # noqa: BLE001
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

    # Records to reprocess (will be populated based on conditions)
    records_to_reprocess = []

    # Handle error records if they exist
    if error_records is not None and not error_records.empty:
        # Extract identifiers from error records
        error_identifiers: list[Union[DataValue, tuple]] = []
        if isinstance(right_on or on, str):
            error_identifiers = list(error_records.collect(str(right_on or on)))
        else:
            # For multiple fields, collect a list of tuples
            error_identifiers = list(error_records.collect(*(right_on or on)))

        # Filter source dataset for records matching error identifiers
        if isinstance(on, str):
            error_source_records = dc.filter(C(on).isin(error_identifiers))
        else:
            # For multiple fields, we need to create compound conditions
            conditions = []
            for i, field in enumerate(on):
                # Each condition checks if the field matches any value
                # from error_identifiers
                field_values = [
                    identifier[i] if isinstance(identifier, tuple) else identifier
                    for identifier in error_identifiers
                ]
                conditions.append(C(field).isin(field_values))

            # Combine conditions with AND
            from functools import reduce

            error_source_records = dc.filter(reduce(lambda x, y: x & y, conditions))

        records_to_reprocess.append(error_source_records)

    # Handle missing records if retry_missing is True
    if retry_missing:
        # Create a subtract operation to find missing records
        missing_records = dc.subtract(result_dataset, on=on, right_on=right_on)

        if not missing_records.empty:
            records_to_reprocess.append(missing_records)

    # If no records to reprocess, return None
    if not records_to_reprocess:
        return None, False

    # Combine all records to reprocess if there are multiple sources
    if len(records_to_reprocess) > 1:
        from functools import reduce

        reprocess_chain = reduce(lambda x, y: x.union(y), records_to_reprocess)
    elif len(records_to_reprocess) == 1:
        reprocess_chain = records_to_reprocess[0]
    else:
        return None, False

    # Return the chain with records to reprocess
    return reprocess_chain, True
