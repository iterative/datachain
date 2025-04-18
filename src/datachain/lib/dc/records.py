from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional, Union

import sqlalchemy

from datachain.lib.data_model import DataType
from datachain.lib.file import (
    File,
)
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session
from datachain.query.schema import Column

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_records(
    to_insert: Optional[Union[dict, Iterable[dict]]],
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    schema: Optional[dict[str, DataType]] = None,
) -> "DataChain":
    """Create a DataChain from the provided records. This method can be used for
    programmatically generating a chain in contrast of reading data from storages
    or other sources.

    Parameters:
        to_insert : records (or a single record) to insert. Each record is
                    a dictionary of signals and theirs values.
        schema : describes chain signals and their corresponding types

    Example:
        ```py
        import datachain as dc
        single_record = dc.read_records(dc.DEFAULT_FILE_RECORD)
        ```
    """
    from .datasets import read_dataset

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    name = session.generate_temp_dataset_name()
    signal_schema = None
    columns: list[sqlalchemy.Column] = []

    if schema:
        signal_schema = SignalSchema(schema)
        columns = []
        for c in signal_schema.db_signals(as_columns=True):
            assert isinstance(c, Column)
            kw = {"nullable": c.nullable} if c.nullable is not None else {}
            columns.append(sqlalchemy.Column(c.name, c.type, **kw))
    else:
        columns = [
            sqlalchemy.Column(name, typ)
            for name, typ in File._datachain_column_types.items()
        ]

    dsr = catalog.create_dataset(
        name,
        columns=columns,
        feature_schema=(
            signal_schema.clone_without_sys_signals().serialize()
            if signal_schema
            else None
        ),
    )

    session.add_dataset_version(dsr, dsr.latest_version)

    if isinstance(to_insert, dict):
        to_insert = [to_insert]
    elif not to_insert:
        to_insert = []

    warehouse = catalog.warehouse
    dr = warehouse.dataset_rows(dsr)
    table = dr.get_table()
    warehouse.insert_rows(table, to_insert)
    warehouse.insert_rows_done(table)
    return read_dataset(name=dsr.name, session=session, settings=settings)
