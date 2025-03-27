import os.path
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

from datachain.lib.file import (
    FileType,
    get_file_type,
)
from datachain.lib.listing import get_file_info, get_listing, list_bucket, ls
from datachain.query import Session

if TYPE_CHECKING:
    from .datachain import DataChain


def from_storage(
    uri: Union[str, os.PathLike[str]],
    *,
    type: FileType = "binary",
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    recursive: Optional[bool] = True,
    object_name: str = "file",
    update: bool = False,
    anon: bool = False,
    client_config: Optional[dict] = None,
) -> "DataChain":
    """Get data from a storage as a list of file with all file attributes.
    It returns the chain itself as usual.

    Parameters:
        uri : storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///"
        type : read file as "binary", "text", or "image" data. Default is "binary".
        recursive : search recursively for the given path.
        object_name : Created object column name.
        update : force storage reindexing. Default is False.
        anon : If True, we will treat cloud bucket as public one
        client_config : Optional client configuration for the storage client.

    Example:
        Simple call from s3
        ```py
        import datachain as dc
        chain = dc.from_storage("s3://my-bucket/my-dir")
        ```

        With AWS S3-compatible storage
        ```py
        import datachain as dc
        chain = dc.from_storage(
            "s3://my-bucket/my-dir",
            client_config = {"aws_endpoint_url": "<minio-endpoint-url>"}
        )
        ```

        Pass existing session
        ```py
        session = Session.get()
        import datachain as dc
        chain = dc.from_storage("s3://my-bucket/my-dir", session=session)
        ```
    """
    from .datachain import DataChain
    from .datasets import from_dataset
    from .records import from_records
    from .values import from_values

    file_type = get_file_type(type)

    if anon:
        client_config = (client_config or {}) | {"anon": True}
    session = Session.get(session, client_config=client_config, in_memory=in_memory)
    cache = session.catalog.cache
    client_config = session.catalog.client_config

    list_ds_name, list_uri, list_path, list_ds_exists = get_listing(
        uri, session, update=update
    )

    # ds_name is None if object is a file, we don't want to use cache
    # or do listing in that case - just read that single object
    if not list_ds_name:
        dc = from_values(
            session=session,
            settings=settings,
            in_memory=in_memory,
            file=[get_file_info(list_uri, cache, client_config=client_config)],
        )
        dc.signals_schema = dc.signals_schema.mutate({f"{object_name}": file_type})
        return dc

    dc = from_dataset(list_ds_name, session=session, settings=settings)
    dc.signals_schema = dc.signals_schema.mutate({f"{object_name}": file_type})

    if update or not list_ds_exists:

        def lst_fn():
            # disable prefetch for listing, as it pre-downloads all files
            (
                from_records(
                    DataChain.DEFAULT_FILE_RECORD,
                    session=session,
                    settings=settings,
                    in_memory=in_memory,
                )
                .settings(prefetch=0)
                .gen(
                    list_bucket(list_uri, cache, client_config=client_config),
                    output={f"{object_name}": file_type},
                )
                .save(list_ds_name, listing=True)
            )

        dc._query.add_before_steps(lst_fn)

    return ls(dc, list_path, recursive=recursive, object_name=object_name)
