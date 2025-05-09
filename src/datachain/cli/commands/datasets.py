import sys
from typing import TYPE_CHECKING, Optional

from tabulate import tabulate

if TYPE_CHECKING:
    from datachain.catalog import Catalog

from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.error import DatasetNotFoundError
from datachain.studio import list_datasets as list_datasets_studio


def group_dataset_versions(datasets, latest_only=True):
    grouped = {}
    # Sort to ensure groupby works as expected
    # (groupby expects consecutive items with the same key)
    for name, version in sorted(datasets):
        grouped.setdefault(name, []).append(version)

    if latest_only:
        # For each dataset name, pick the highest version.
        return {name: max(versions) for name, versions in grouped.items()}
    # For each dataset name, return a sorted list of unique versions.
    return {name: sorted(set(versions)) for name, versions in grouped.items()}


def list_datasets(
    catalog: "Catalog",
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
    latest_only: bool = True,
    name: Optional[str] = None,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)
    if name:
        latest_only = False

    local_datasets = set(list_datasets_local(catalog, name)) if all or local else set()
    studio_datasets = (
        set(list_datasets_studio(team=team, name=name))
        if (all or studio) and token
        else set()
    )

    # Group the datasets for both local and studio sources.
    local_grouped = group_dataset_versions(local_datasets, latest_only)
    studio_grouped = group_dataset_versions(studio_datasets, latest_only)

    # Merge all dataset names from both sources.
    all_dataset_names = sorted(set(local_grouped.keys()) | set(studio_grouped.keys()))

    datasets = []
    if latest_only:
        # For each dataset name, get the latest version from each source (if available).
        for n in all_dataset_names:
            datasets.append((n, (local_grouped.get(n), studio_grouped.get(n))))
    else:
        # For each dataset name, merge all versions from both sources.
        for n in all_dataset_names:
            local_versions = local_grouped.get(n, [])
            studio_versions = studio_grouped.get(n, [])

            # If neither source has any versions, record it as (None, None).
            if not local_versions and not studio_versions:
                datasets.append((n, (None, None)))
            else:
                # For each unique version from either source, record its presence.
                for version in sorted(set(local_versions) | set(studio_versions)):
                    datasets.append(
                        (
                            n,
                            (
                                version if version in local_versions else None,
                                version if version in studio_versions else None,
                            ),
                        )
                    )

    rows = [
        _datasets_tabulate_row(
            name=n,
            both=(all or (local and studio)) and token,
            local_version=local_version,
            studio_version=studio_version,
        )
        for n, (local_version, studio_version) in datasets
    ]

    print(tabulate(rows, headers="keys"))


def list_datasets_local(catalog: "Catalog", name: Optional[str] = None):
    if name:
        yield from list_datasets_local_versions(catalog, name)
        return

    for d in catalog.ls_datasets():
        for v in d.versions:
            yield (d.name, v.version)


def list_datasets_local_versions(catalog: "Catalog", name: str):
    ds = catalog.get_dataset(name)
    for v in ds.versions:
        yield (name, v.version)


def _datasets_tabulate_row(name, both, local_version, studio_version):
    row = {
        "Name": name,
    }
    if both:
        row["Studio"] = f"v{studio_version}" if studio_version else "\u2716"
        row["Local"] = f"v{local_version}" if local_version else "\u2716"
    else:
        latest_version = local_version or studio_version
        row["Latest Version"] = f"v{latest_version}" if latest_version else "\u2716"

    return row


def rm_dataset(
    catalog: "Catalog",
    name: str,
    version: Optional[str] = None,
    force: Optional[bool] = False,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    from datachain.studio import remove_studio_dataset

    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)

    if all or local:
        try:
            catalog.remove_dataset(name, version=version, force=force)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)

    if (all or studio) and token:
        remove_studio_dataset(team, name, version, force)


def edit_dataset(
    catalog: "Catalog",
    name: str,
    new_name: Optional[str] = None,
    description: Optional[str] = None,
    attrs: Optional[list[str]] = None,
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    from datachain.studio import edit_studio_dataset

    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)

    if all or local:
        try:
            catalog.edit_dataset(name, new_name, description, attrs)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)

    if (all or studio) and token:
        edit_studio_dataset(team, name, new_name, description, attrs)
