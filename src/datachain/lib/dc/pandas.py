from typing import (
    TYPE_CHECKING,
    Optional,
)

from datachain.query import Session

from .values import from_values

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def from_pandas(  # type: ignore[override]
    df: "pd.DataFrame",
    name: str = "",
    session: Optional[Session] = None,
    settings: Optional[dict] = None,
    in_memory: bool = False,
    object_name: str = "",
) -> "DataChain":
    """Generate chain from pandas data-frame.

    Example:
        ```py
        import pandas as pd
        import datachain as dc

        df = pd.DataFrame({"fib": [1, 2, 3, 5, 8]})
        dc.from_pandas(df)
        ```
    """
    from .utils import DatasetPrepareError

    fr_map = {col.lower(): df[col].tolist() for col in df.columns}

    for column in fr_map:
        if not column.isidentifier():
            raise DatasetPrepareError(
                name,
                f"import from pandas error - '{column}' cannot be a column name",
            )

    return from_values(
        name,
        session,
        settings=settings,
        object_name=object_name,
        in_memory=in_memory,
        **fr_map,
    )
