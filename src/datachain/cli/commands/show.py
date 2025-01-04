from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def show(
    catalog: "Catalog",
    name: str,
    version: Optional[int] = None,
    limit: int = 10,
    offset: int = 0,
    columns: Sequence[str] = (),
    no_collapse: bool = False,
    schema: bool = False,
) -> None:
    from datachain import Session
    from datachain.lib.dc import DataChain
    from datachain.query.dataset import DatasetQuery
    from datachain.utils import show_records

    dataset = catalog.get_dataset(name)
    dataset_version = dataset.get_version(version or dataset.latest_version)

    query = (
        DatasetQuery(name=name, version=version, catalog=catalog)
        .select(*columns)
        .limit(limit)
        .offset(offset)
    )
    records = query.to_db_records()
    show_records(records, collapse_columns=not no_collapse)
    if schema and dataset_version.feature_schema:
        print("\nSchema:")
        session = Session.get(catalog=catalog)
        dc = DataChain.from_dataset(name=name, version=version, session=session)
        dc.print_schema()
