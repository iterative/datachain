import dataclasses
import io
import math
import os
import tarfile
from string import printable
from tarfile import DIRTYPE, TarInfo
from time import sleep, time
from typing import Any, Callable, Optional

import pytest

from datachain.catalog.catalog import Catalog
from datachain.dataset import DatasetDependency, DatasetRecord
from datachain.query import C, DatasetQuery
from datachain.query.builtins import index_tar
from datachain.storage import StorageStatus


def make_index(catalog, src: str, entries, ttl: int = 1234):
    lst, _ = catalog.enlist_source(src, ttl, skip_indexing=True)
    lst.insert_entries(entries)
    lst.insert_entries_done()
    lst.metastore.mark_storage_indexed(
        src,
        StorageStatus.COMPLETE,
        ttl=ttl,
        prefix="",
        partial_id=lst.metastore.partial_id,
    )


DEFAULT_TREE: dict[str, Any] = {
    "description": "Cats and Dogs",
    "cats": {
        "cat1": "meow",
        "cat2": "mrow",
    },
    "dogs": {
        "dog1": "woof",
        "dog2": "arf",
        "dog3": "bark",
        "others": {"dog4": "ruff"},
    },
}
NUM_TREE = {f"{i:06d}": f"{i}" for i in range(1024)}


def instantiate_tree(path, tree):
    for key, value in tree.items():
        if isinstance(value, str):
            (path / key).write_text(value)
        elif isinstance(value, bytes):
            (path / key).write_bytes(value)
        elif isinstance(value, dict):
            (path / key).mkdir()
            instantiate_tree(path / key, value)
        else:
            raise TypeError(f"{value=}")


def tree_from_path(path, binary=False):
    tree = {}
    for child in path.iterdir():
        if child.is_dir():
            tree[child.name] = tree_from_path(child, binary)
        else:  # noqa: PLR5501
            if binary:
                tree[child.name] = child.read_bytes()
            else:
                tree[child.name] = child.read_text()
    return tree


def uppercase_scheme(uri: str) -> str:
    """
    Makes scheme (or protocol) of an url uppercased
    e.g s3://bucket_name -> S3://bucket_name
    """
    return f'{uri.split(":")[0].upper()}:{":".join(uri.split(":")[1:])}'


def make_tar(tree) -> bytes:
    with io.BytesIO() as tmp:
        with tarfile.open(fileobj=tmp, mode="w") as archive:
            write_tar(tree, archive)
        return tmp.getvalue()


def write_tar(tree, archive, curr_dir=""):
    for key, value in tree.items():
        name = f"{curr_dir}/{key}" if curr_dir else key
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            info = TarInfo(name)
            info.size = len(value)
            f = io.BytesIO(value)
            archive.addfile(info, f)
        elif isinstance(value, dict):
            info = TarInfo(name)
            info.type = DIRTYPE
            archive.addfile(info, io.BytesIO())
            write_tar(value, archive, name)


TARRED_TREE: dict[str, Any] = {"animals.tar": make_tar(DEFAULT_TREE)}


def create_tar_dataset(catalog, uri: str, ds_name: str) -> DatasetQuery:
    """
    Create a dataset from a storage location containing tar archives and other files.

    The resulting dataset contains both the original files (as regular objects)
    and the tar members (as v-objects).
    """
    ds1 = DatasetQuery(path=uri, catalog=catalog)
    tar_entries = ds1.filter(C("name").glob("*.tar")).generate(index_tar)
    return ds1.filter(~C("name").glob("*.tar")).union(tar_entries).save(ds_name)


def skip_if_not_sqlite():
    if os.environ.get("DATACHAIN_METASTORE") or os.environ.get("DATACHAIN_WAREHOUSE"):
        pytest.skip("This test is not supported on other data storages")


WEBFORMAT_TREE: dict[str, Any] = {
    "f1.raw": "raw data",
    "f1.json": '{"similarity": 0.001, "md5": "deadbeef"}',
    "f2.raw": "raw data",
    "f2.json": '{"similarity": 0.005, "md5": "foobar"}',
}


def text_embedding(text: str) -> list[float]:
    """
    Compute a simple text embedding based on character counts.

    These aren't the most meaningful, but will produce a 100-element
    vector of floats between 0 and 1 where texts with similar
    character counts will have similar embeddings. Useful for writing
    unit tests without loading a heavy ML model.
    """
    emb = dict.fromkeys(printable, 0.01)
    for c in text:
        try:
            emb[c] += 1.0
        except KeyError:
            pass
    # sqeeze values between 0 and 1 with an adjusted sigmoid function
    return [2.0 / (1.0 + math.e ** (-x)) - 1.0 for x in emb.values()]


SIMPLE_DS_QUERY_RECORDS = [
    {
        "parent": "",
        "name": "description",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 13,
    },
    {
        "parent": "cats",
        "name": "cat1",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 4,
    },
    {
        "parent": "cats",
        "name": "cat2",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 4,
    },
    {
        "parent": "dogs",
        "name": "dog1",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 4,
    },
    {
        "parent": "dogs",
        "name": "dog2",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 3,
    },
    {
        "parent": "dogs",
        "name": "dog3",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 4,
    },
    {
        "parent": "dogs/others",
        "name": "dog4",
        "vtype": "",
        "dir_type": 0,
        "is_latest": 1,
        "size": 4,
    },
]


def get_simple_ds_query(path, catalog):
    return (
        DatasetQuery(path=path, catalog=catalog)
        .select(C.parent, C.name, C.vtype, C.dir_type, C.is_latest, C.size)
        .order_by(C.source, C.parent, C.name)
    )


def dataset_dependency_asdict(
    dep: Optional[DatasetDependency],
) -> Optional[dict[str, Any]]:
    """
    Converting to dict with making sure we don't have any additional fields
    that could've been added with subclasses
    """
    if not dep:
        return None
    parsed = {
        k: dataclasses.asdict(dep)[k] for k in DatasetDependency.__dataclass_fields__
    }

    if dep.dependencies:
        parsed["dependencies"] = [
            dataset_dependency_asdict(d) for d in dep.dependencies
        ]

    return parsed


def wait_for_condition(
    callback: Callable,
    message: str,
    check_interval: float = 0.01,
    timeout: float = 1.0,
) -> Any:
    start_time = time()
    while time() - start_time < timeout:
        result = callback()
        if result:
            return result
        sleep(check_interval)
    raise TimeoutError(f"Timeout expired while waiting for: {message}")


def assert_row_names(
    catalog: Catalog, dataset: DatasetRecord, version: int, expected_names: set
) -> None:
    dataset_rows = catalog.ls_dataset_rows(dataset.name, version, limit=20)
    assert dataset_rows
    preview = dataset.get_version(version).preview
    assert preview

    assert (
        {r["name"] for r in dataset_rows}
        == {r.get("name") for r in preview}
        == expected_names
    )
