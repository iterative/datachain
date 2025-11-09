from typing import TYPE_CHECKING

import shtab

if TYPE_CHECKING:
    from datachain.catalog import Catalog


def clear_cache(catalog: "Catalog"):
    catalog.cache.clear()


def garbage_collect(catalog: "Catalog"):
    temp_tables = catalog.get_temp_table_names()
    has_tables = bool(temp_tables)

    if has_tables:
        print(f"Garbage collecting {len(temp_tables)} temporary tables.")
        catalog.cleanup_tables(temp_tables)

    if not has_tables:
        print("No temporary tables to clean up.")


def completion(shell: str) -> str:
    from datachain.cli import get_parser

    return shtab.complete(
        get_parser(),
        shell=shell,
    )
