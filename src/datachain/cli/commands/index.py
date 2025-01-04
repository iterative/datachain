from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def index(
    catalog: "Catalog",
    sources,
    **kwargs,
):
    catalog.index(sources, **kwargs)
