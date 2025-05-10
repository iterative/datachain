import os.path
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from datachain.error import DatasetNotFoundError
from datachain.lib.file import (
    FileType,
    get_file_type,
)
from datachain.lib.listing import (
    get_file_info,
    get_listing,
    list_bucket,
    ls,
)
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def read_storage(
    uri: Union[str, os.PathLike[str], list[str], list[os.PathLike[str]]],
    *,
    type: FileType = "binary",
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    recursive: Optional[bool] = True,
    column: str = "file",
    update: bool = False,
    anon: bool = False,
    client_config: Optional[dict] = None,
) -> "DataChain":
    """Get data from storage(s) as a list of file with all file attributes.
    It returns the chain itself as usual.

    Parameters:
        uri : storage URI with directory or list of URIs.
            URIs must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///"
        type : read file as "binary", "text", or "image" data. Default is "binary".
        recursive : search recursively for the given path.
        column : Created column name.
        update : force storage reindexing. Default is False.
        anon : If True, we will treat cloud bucket as public one
        client_config : Optional client configuration for the storage client.

    Returns:
        DataChain: A DataChain object containing the file information.

    Examples:
        Simple call from s3:
        ```python
        import datachain as dc
        chain = dc.read_storage("s3://my-bucket/my-dir")
        ```

        Multiple URIs:
        ```python
        chain = dc.read_storage([
            "s3://bucket1/dir1",
            "s3://bucket2/dir2"
        ])
        ```

        With AWS S3-compatible storage:
        ```python
        chain = dc.read_storage(
            "s3://my-bucket/my-dir",
            client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
        )
        ```

        Pass existing session
        ```py
        session = Session.get()
        chain = dc.read_storage([
            "path/to/dir1",
            "path/to/dir2"
        ], session=session, recursive=True)
        ```

    Note:
        When using multiple URIs with `update=True`, the function optimizes by
        avoiding redundant updates for URIs pointing to the same storage location.
    """
    from .datachain import DataChain
    from .datasets import read_dataset
    from .records import read_records
    from .values import read_values

    file_type = get_file_type(type)

    if anon:
        client_config = (client_config or {}) | {"anon": True}
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    catalog = session.catalog
    cache = catalog.cache
    client_config = session.catalog.client_config

    uris = uri if isinstance(uri, (list, tuple)) else [uri]

    if not uris:
        raise ValueError("No URIs provided")

    storage_chain = None
    listed_ds_name = set()
    file_values = []

    for single_uri in uris:
        list_ds_name, list_uri, list_path, list_ds_exists = get_listing(
            single_uri, session, update=update
        )

        # list_ds_name is None if object is a file, we don't want to use cache
        # or do listing in that case - just read that single object
        if not list_ds_name:
            file_values.append(
                get_file_info(list_uri, cache, client_config=client_config)
            )
            continue

        dc = read_dataset(list_ds_name, session=session, settings=settings)
        dc._query.update = update
        dc.signals_schema = dc.signals_schema.mutate({f"{column}": file_type})

        if update or not list_ds_exists:

            def lst_fn(ds_name, lst_uri):
                # disable prefetch for listing, as it pre-downloads all files
                try:
                    version = catalog.get_dataset(ds_name).next_version_major
                except DatasetNotFoundError:
                    version = None

                (
                    read_records(
                        DataChain.DEFAULT_FILE_RECORD,
                        session=session,
                        settings=settings,
                        in_memory=in_memory,
                    )
                    .settings(prefetch=0)
                    .gen(
                        list_bucket(lst_uri, cache, client_config=client_config),
                        output={f"{column}": file_type},
                    )
                    # for internal listing datasets, we always bump major version
                    .save(ds_name, listing=True, version=version)
                )

            dc._query.set_listing_fn(
                lambda ds_name=list_ds_name, lst_uri=list_uri: lst_fn(ds_name, lst_uri)
            )

        chain = ls(dc, list_path, recursive=recursive, column=column)

        storage_chain = storage_chain.union(chain) if storage_chain else chain
        listed_ds_name.add(list_ds_name)

    if file_values:
        file_chain = read_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            file=file_values,
        )
        file_chain.signals_schema = file_chain.signals_schema.mutate(
            {f"{column}": file_type}
        )
        storage_chain = storage_chain.union(file_chain) if storage_chain else file_chain

    assert storage_chain is not None

    return storage_chain
