from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union, get_origin, get_type_hints

from datachain.error import DatasetVersionNotFoundError
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.file import (
    File,
)
from datachain.lib.settings import Settings
from datachain.lib.signal_schema import SignalSchema
from datachain.query import Session
from datachain.query.dataset import DatasetQuery

from .utils import Sys
from .values import read_values

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_dataset(
    name: str,
    version: Optional[Union[str, int]] = None,
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    fallback_to_studio: bool = True,
    delta: Optional[bool] = False,
    delta_on: Optional[Union[str, Sequence[str]]] = None,
    delta_result_on: Optional[Union[str, Sequence[str]]] = None,
    delta_compare: Optional[Union[str, Sequence[str]]] = None,
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
        delta: If set to True, we optimize the creation of new dataset versions by
            calculating the diff between the latest version of this storage and the
            version used to create the most recent version of the resulting chain
            dataset (the one specified in `.save()`). We then run the "diff" chain
            using only the diff data, rather than the entire storage data, and merge
            that diff chain with the latest version of the resulting dataset to create
            a new version. This approach avoids applying modifications to all records
            from storage every time, which can be an expensive operation.
            The diff is calculated using the `DataChain.compare()` method, which
            compares the `delta_on` fields to find matches and checks the compare
            fields to determine if a record has changed. Note that this process only
            considers added and modified records in storage; deleted records are not
            removed from the new dataset version.
            This calculation is based on the difference between the current version
            of the source and the version used to create the dataset.
        delta_on: A list of fields that uniquely identify rows in the source.
            If two rows have the same values, they are considered the same (e.g., they
            could be different versions of the same row in a versioned source).
            This is used in the delta update to calculate the diff.
        delta_result_on: A list of fields in the resulting dataset that correspond
            to the `delta_on` fields from the source.
            This is needed to identify rows that have changed in the source but are
            already present in the current version of the resulting dataset, in order
            to avoid including outdated versions of those rows in the new dataset.
            We retain only the latest versions of rows to prevent duplication.
            There is no need to define this if the `delta_on` fields are present in
            the final dataset and have not been renamed.
        delta_compare: A list of fields used to check if the same row has been modified
            in the new version of the source.
            If not defined, all fields except those defined in delta_on will be used.

    Example:
        ```py
        import datachain as dc
        chain = dc.read_dataset("my_cats")
        ```

        ```py
        chain = dc.read_dataset("my_cats", fallback_to_studio=False)
        ```

        ```py
        chain = dc.read_dataset("my_cats", version="1.0.0")
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
        chain = dc.read_dataset(
            name="my_cats",
            version="1.0.0",
            session=session,
            settings=settings,
            fallback_to_studio=True,
        )
        ```
    """
    from datachain.telemetry import telemetry

    from .datachain import DataChain

    if version is not None:
        try:
            # for backward compatibility we still allow users to put version as integer
            # in which case we are trying to find latest version where major part is
            # equal to that input version. For example if user sets version=2, we could
            # continue with something like 2.4.3 (assuming 2.4.3 is the biggest among
            # all 2.* dataset versions). If dataset doesn't have any versions where
            # major part is equal to that input, exception is thrown.
            major = int(version)
            dataset = Session.get(session).catalog.get_dataset(name)
            latest_major = dataset.latest_major_version(major)
            if not latest_major:
                raise DatasetVersionNotFoundError(
                    f"Dataset {name} does not have version {version}"
                )
            version = latest_major
        except ValueError:
            # version is in new semver string format, continuing as normal
            pass

    query = DatasetQuery(
        name=name,
        version=version,  #  type: ignore[arg-type]
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
    chain = DataChain(query, _settings, signals_schema)
    if delta:
        chain = chain._as_delta(
            on=delta_on, right_on=delta_result_on, compare=delta_compare
        )
    return chain


def datasets(
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    column: Optional[str] = None,
    include_listing: bool = False,
    studio: bool = False,
    attrs: Optional[list[str]] = None,
) -> "DataChain":
    """Generate chain with list of registered datasets.

    Args:
        session: Optional session instance. If not provided, uses default session.
        settings: Optional dictionary of settings to configure the chain.
        in_memory: If True, creates an in-memory session. Defaults to False.
        column: Name of the output column in the chain. Defaults to None which
            means no top level column will be created.
        include_listing: If True, includes listing datasets. Defaults to False.
        studio: If True, returns datasets from Studio only,
            otherwise returns all local datasets. Defaults to False.
        attrs: Optional list of attributes to filter datasets on. It can be just
            attribute without value e.g "NLP", or attribute with value
            e.g "location=US". Attribute with value can also accept "*" to target
            all that have specific name e.g "location=*"

    Returns:
        DataChain: A new DataChain instance containing dataset information.

    Example:
        ```py
        import datachain as dc

        chain = dc.datasets(column="dataset")
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
    datasets_values = [d for d in datasets_values if not d.is_temp]

    if attrs:
        for attr in attrs:
            datasets_values = [d for d in datasets_values if d.has_attr(attr)]

    if not column:
        # flattening dataset fields
        schema = {
            k: get_origin(v) if get_origin(v) is dict else v
            for k, v in get_type_hints(DatasetInfo).items()
            if k in DatasetInfo.model_fields
        }
        data = {k: [] for k in DatasetInfo.model_fields}  # type: ignore[var-annotated]
        for d in [d.model_dump() for d in datasets_values]:
            for field, value in d.items():
                data[field].append(value)

        return read_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            output=schema,
            **data,  # type: ignore[arg-type]
        )

    return read_values(
        session=session,
        settings=settings,
        in_memory=in_memory,
        output={column: DatasetInfo},
        **{column: datasets_values},  # type: ignore[arg-type]
    )


def delete_dataset(
    name: str,
    version: Optional[str] = None,
    force: Optional[bool] = False,
    studio: Optional[bool] = False,
    session: Optional[Session] = None,
    in_memory: bool = False,
) -> None:
    """Removes specific dataset version or all dataset versions, depending on
    a force flag.

    Args:
        name : Dataset name
        version : Optional dataset version
        force: If true, all datasets versions will be removed. Defaults to False.
        studio: If True, removes dataset from Studio only,
            otherwise remove from local. Defaults to False.
        session: Optional session instance. If not provided, uses default session.
        in_memory: If True, creates an in-memory session. Defaults to False.

    Returns: None

    Example:
        ```py
        import datachain as dc
        dc.delete_dataset("cats")
        ```

        ```py
        import datachain as dc
        dc.delete_dataset("cats", version="1.0.0")
        ```
    """

    session = Session.get(session, in_memory=in_memory)
    catalog = session.catalog
    if not force:
        version = version or catalog.get_dataset(name).latest_version
    else:
        version = None
    catalog.remove_dataset(name, version=version, force=force, studio=studio)
