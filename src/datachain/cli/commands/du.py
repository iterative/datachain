from typing import TYPE_CHECKING

from datachain import utils

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def du(catalog: "Catalog", sources, show_bytes=False, si=False, **kwargs):
    for path, size in catalog.du(sources, **kwargs):
        if show_bytes:
            print(f"{size} {path}")
        else:
            print(f"{utils.sizeof_fmt(size, si=si): >7} {path}")
