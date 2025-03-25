from typing import (
    TYPE_CHECKING,
    Optional,
)

from datachain.lib.listing_info import ListingInfo
from datachain.query import Session

from .values import from_values

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from .datachain import DataChain

    P = ParamSpec("P")


def listings(
    session: Optional[Session] = None,
    in_memory: bool = False,
    object_name: str = "listing",
    **kwargs,
) -> "DataChain":
    """Generate chain with list of cached listings.
    Listing is a special kind of dataset which has directory listing data of
    some underlying storage (e.g S3 bucket).

    Example:
        ```py
        import datachain as dc
        dc.listings().show()
        ```
    """
    session = Session.get(session, in_memory=in_memory)
    catalog = kwargs.get("catalog") or session.catalog

    return from_values(
        session=session,
        in_memory=in_memory,
        output={object_name: ListingInfo},
        **{object_name: catalog.listings()},  # type: ignore[arg-type]
    )
