from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

from datachain.lib.data_model import DataType
from datachain.query import Session

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def read_parquet(
    path,
    partitioning: Any = "hive",
    output: Optional[dict[str, DataType]] = None,
    column: str = "",
    model_name: str = "",
    source: bool = True,
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    **kwargs,
) -> "DataChain":
    """Generate chain from parquet files.

    Parameters:
        path : Storage URI with directory. URI must start with storage prefix such
            as `s3://`, `gs://`, `az://` or "file:///".
        partitioning : Any pyarrow partitioning schema.
        output : Dictionary defining column names and their corresponding types.
        column : Created column name.
        model_name : Generated model name.
        source : Whether to include info about the source file.
        session : Session to use for the chain.
        settings : Settings to use for the chain.

    Example:
        Reading a single file:
        ```py
        import datachain as dc
        dc.read_parquet("s3://mybucket/file.parquet")
        ```

        Reading a partitioned dataset from a directory:
        ```py
        import datachain as dc
        dc.read_parquet("s3://mybucket/dir")
        ```
    """
    from .storage import read_storage

    chain = read_storage(path, session=session, settings=settings, **kwargs)
    return chain.parse_tabular(
        output=output,
        column=column,
        model_name=model_name,
        source=source,
        format="parquet",
        partitioning=partitioning,
    )
