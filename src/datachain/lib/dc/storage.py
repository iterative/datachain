import os.path
from collections.abc import Sequence
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from datachain.lib.dc.storage_pattern import (
    convert_globstar_to_sqlite,
    expand_uri_braces,
    should_use_recursion,
    split_uri_pattern,
)
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


def _apply_glob_filter(
    dc: "DataChain",
    patterns: list[str],
    list_path: str,
    use_recursive: bool,
    column: str,
) -> "DataChain":
    """Apply glob filter to a DataChain based on a single pattern.

    Since brace expansion now happens at URI level, this function
    only needs to handle single patterns.
    """
    from datachain.query.schema import Column

    chain = ls(dc, list_path, recursive=use_recursive, column=column)

    # Should only receive single patterns now (brace expansion happens earlier)
    if len(patterns) != 1:
        raise ValueError(f"Expected single pattern, got {len(patterns)}")

    pattern = patterns[0]

    # If pattern doesn't contain path separator and list_path is not empty,
    # prepend the list_path to make the pattern match correctly
    if list_path and "/" not in pattern:
        filter_pattern = f"{list_path.rstrip('/')}/{pattern}"
    else:
        filter_pattern = pattern

    # Convert globstar patterns to SQLite-compatible patterns
    sqlite_pattern = convert_globstar_to_sqlite(filter_pattern)

    return chain.filter(Column(f"{column}.path").glob(sqlite_pattern))



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
    anon: Optional[bool] = None,
    delta: Optional[bool] = False,
    delta_on: Optional[Union[str, Sequence[str]]] = (
        "file.path",
        "file.etag",
        "file.version",
    ),
    delta_result_on: Optional[Union[str, Sequence[str]]] = None,
    delta_compare: Optional[Union[str, Sequence[str]]] = None,
    delta_retry: Optional[Union[bool, str]] = None,
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
        delta: If True, only process new or changed files instead of reprocessing
            everything. This saves time by skipping files that were already processed in
            previous versions. The optimization is working when a new version of the
            dataset is created.
            Default is False.
        delta_on: Field(s) that uniquely identify each record in the source data.
            Used to detect which records are new or changed.
            Default is ("file.path", "file.etag", "file.version").
        delta_result_on: Field(s) in the result dataset that match `delta_on` fields.
            Only needed if you rename the identifying fields during processing.
            Default is None.
        delta_compare: Field(s) used to detect if a record has changed.
            If not specified, all fields except `delta_on` fields are used.
            Default is None.
        delta_retry: Controls retry behavior for failed records:
            - String (field name): Reprocess records where this field is not empty
              (error mode)
            - True: Reprocess records missing from the result dataset (missing mode)
            - None: No retry processing (default)

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

    if anon is not None:
        client_config = (client_config or {}) | {"anon": anon}
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    catalog = session.catalog
    cache = catalog.cache
    client_config = session.catalog.client_config
    listing_namespace_name = catalog.metastore.system_namespace_name
    listing_project_name = catalog.metastore.listing_project_name

    uris = uri if isinstance(uri, (list, tuple)) else [uri]

    if not uris:
        raise ValueError("No URIs provided")

    # First, expand all URIs that contain brace patterns
    expanded_uris = []
    for single_uri in uris:
        uri_str = str(single_uri)
        # Check if URI contains braces and expand them
        if "{" in uri_str and "}" in uri_str:
            expanded_uris.extend(expand_uri_braces(uri_str))
        else:
            expanded_uris.append(uri_str)

    # Now process each expanded URI
    chains = []
    listed_ds_name = set()
    file_values = []

    for single_uri in expanded_uris:
        # Check if URI contains glob patterns and split them
        base_uri, glob_pattern = split_uri_pattern(single_uri)

        # If a pattern is found, use the base_uri for listing
        # The pattern will be used for filtering later
        list_uri_to_use = base_uri if glob_pattern else single_uri

        list_ds_name, list_uri, list_path, list_ds_exists = get_listing(
            list_uri_to_use, session, update=update
        )

        # list_ds_name is None if object is a file, we don't want to use cache
        # or do listing in that case - just read that single object
        if not list_ds_name:
            file_values.append(
                get_file_info(list_uri, cache, client_config=client_config)
            )
            continue

        dc = read_dataset(
            list_ds_name,
            namespace=listing_namespace_name,
            project=listing_project_name,
            session=session,
            settings=settings,
        )
        dc._query.update = update
        dc.signals_schema = dc.signals_schema.mutate({f"{column}": file_type})

        if update or not list_ds_exists:

            def lst_fn(ds_name, lst_uri):
                # disable prefetch for listing, as it pre-downloads all files
                (
                    read_records(
                        DataChain.DEFAULT_FILE_RECORD,
                        session=session,
                        settings=settings,
                        in_memory=in_memory,
                    )
                    .settings(
                        prefetch=0,
                        namespace=listing_namespace_name,
                        project=listing_project_name,
                    )
                    .gen(
                        list_bucket(lst_uri, cache, client_config=client_config),
                        output={f"{column}": file_type},
                    )
                    # for internal listing datasets, we always bump major version
                    .save(ds_name, listing=True, update_version="major")
                )

            dc._query.set_listing_fn(
                lambda ds_name=list_ds_name, lst_uri=list_uri: lst_fn(ds_name, lst_uri)
            )

        # If a glob pattern was detected, use it for filtering
        # Otherwise, use the original list_path from get_listing
        if glob_pattern:
            # Determine if we should use recursive listing based on the pattern
            use_recursive = should_use_recursion(glob_pattern, recursive or False)

            # Apply glob filter - no need for brace expansion here as it's done above
            chain = _apply_glob_filter(
                dc, [glob_pattern], list_path, use_recursive, column
            )
            chains.append(chain)
        else:
            # No glob pattern detected, use normal ls behavior
            chains.append(ls(dc, list_path, recursive=recursive, column=column))

        listed_ds_name.add(list_ds_name)

    storage_chain = None if not chains else reduce(lambda x, y: x.union(y), chains)

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

    if delta:
        storage_chain = storage_chain._as_delta(
            on=delta_on,
            right_on=delta_result_on,
            compare=delta_compare,
            delta_retry=delta_retry,
        )

    return storage_chain
