import sys
from typing import TYPE_CHECKING, Optional

from tabulate import tabulate

from datachain import utils

if TYPE_CHECKING:
    from datachain.catalog import Catalog

from datachain.cli.utils import determine_flavors
from datachain.config import Config
from datachain.error import DatasetNotFoundError
from datachain.studio import list_datasets as list_datasets_studio


def list_datasets(
    catalog: "Catalog",
    studio: bool = False,
    local: bool = False,
    all: bool = True,
    team: Optional[str] = None,
):
    token = Config().read().get("studio", {}).get("token")
    all, local, studio = determine_flavors(studio, local, all, token)

    local_datasets = set(list_datasets_local(catalog)) if all or local else set()
    studio_datasets = (
        set(list_datasets_studio(team=team)) if (all or studio) and token else set()
    )

    rows = [
        _datasets_tabulate_row(
            name=name,
            version=version,
            both=(all or (local and studio)) and token,
            local=(name, version) in local_datasets,
            studio=(name, version) in studio_datasets,
        )
        for name, version in local_datasets.union(studio_datasets)
    ]

    print(tabulate(rows, headers="keys"))


def list_datasets_local(catalog: "Catalog"):
    for d in catalog.ls_datasets():
        for v in d.versions:
            yield (d.name, v.version)


def _datasets_tabulate_row(name, version, both, local, studio):
    row = {
        "Name": name,
        "Version": version,
    }
    if both:
        row["Studio"] = "\u2714" if studio else "\u2716"
        row["Local"] = "\u2714" if local else "\u2716"
    return row


def rm_dataset(
    catalog: "Catalog",
    name: str,
    version: Optional[int] = None,
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
    labels: Optional[list[str]] = None,
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
            catalog.edit_dataset(name, new_name, description, labels)
        except DatasetNotFoundError:
            print("Dataset not found in local", file=sys.stderr)

    if (all or studio) and token:
        edit_studio_dataset(team, name, new_name, description, labels)


def dataset_stats(
    catalog: "Catalog",
    name: str,
    version: int,
    show_bytes=False,
    si=False,
):
    stats = catalog.dataset_stats(name, version)

    if stats:
        print(f"Number of objects: {stats.num_objects}")
        if show_bytes:
            print(f"Total objects size: {stats.size}")
        else:
            print(f"Total objects size: {utils.sizeof_fmt(stats.size, si=si): >7}")
