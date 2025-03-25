from typing import (
    TYPE_CHECKING,
    Optional,
)

from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.file import (
    File,
)
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session
from datachain.query.dataset import DatasetQuery

from .utils import Sys
from .values import from_values

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def from_dataset(
    name: str,
    version: Optional[int] = None,
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    fallback_to_studio: bool = True,
) -> "DataChain":
    """Get data from a saved Dataset. It returns the chain itself.
    If dataset or version is not found locally, it will try to pull it from Studio.

    Parameters:
        name : dataset name
        version : dataset version
        session : Session to use for the chain.
        settings : Settings to use for the chain.
        fallback_to_studio : Try to pull dataset from Studio if not found locally.
            Default is True.

    Example:
        ```py
        import datachain as dc
        chain = dc.from_dataset("my_cats")
        ```

        ```py
        chain = dc.from_dataset("my_cats", fallback_to_studio=False)
        ```

        ```py
        chain = dc.from_dataset("my_cats", version=1)
        ```

        ```py
        session = Session.get(client_config={"aws_endpoint_url": "<minio-url>"})
        settings = {
            "cache": True,
            "parallel": 4,
            "workers": 4,
            "min_task_size": 1000,
            "prefetch": 10,
        }
        chain = dc.from_dataset(
            name="my_cats",
            version=1,
            session=session,
            settings=settings,
            fallback_to_studio=True,
        )
        ```
    """
    from datachain.telemetry import telemetry

    from .datachain import DataChain

    query = DatasetQuery(
        name=name,
        version=version,
        session=session,
        indexing_column_types=File._datachain_column_types,
        fallback_to_studio=fallback_to_studio,
    )
    telemetry.send_event_once("class", "datachain_init", name=name, version=version)
    if settings:
        _settings = Settings(**settings)
    else:
        _settings = Settings()

    signals_schema = SignalSchema({"sys": Sys})
    if query.feature_schema:
        signals_schema |= SignalSchema.deserialize(query.feature_schema)
    else:
        signals_schema |= SignalSchema.from_column_types(query.column_types or {})
    return DataChain(query, _settings, signals_schema)


def datasets(
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    object_name: str = "dataset",
    include_listing: bool = False,
    studio: bool = False,
) -> "DataChain":
    """Generate chain with list of registered datasets.

    Args:
        session: Optional session instance. If not provided, uses default session.
        settings: Optional dictionary of settings to configure the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        object_name: Name of the output object in the chain. Defaults to "dataset".
        include_listing: If True, includes listing datasets. Defaults to False.
        studio: If True, returns datasets from Studio only,
            otherwise returns all local datasets. Defaults to False.

    Returns:
        DataChain: A new DataChain instance containing dataset information.

    Example:
        ```py
        import datachain as dc

        chain = dc.datasets()
        for ds in chain.collect("dataset"):
            print(f"{ds.name}@v{ds.version}")
        ```
    """

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog

    datasets_values = [
        DatasetInfo.from_models(d, v, j)
        for d, v, j in catalog.list_datasets_versions(
            include_listing=include_listing, studio=studio
        )
    ]

    return from_values(
        session=session,
        settings=settings,
        in_memory=in_memory,
        output={object_name: DatasetInfo},
        **{object_name: datasets_values},  # type: ignore[arg-type]
    )
