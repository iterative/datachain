import glob
import os
from collections.abc import Iterable, Iterator
from itertools import zip_longest
from typing import TYPE_CHECKING, Optional

from fsspec.asyn import get_loop, sync
from sqlalchemy import Column
from sqlalchemy.sql import func
from tqdm import tqdm

from datachain.node import DirType, Entry, Node, NodeWithPath
from datachain.sql.functions import path as pathfunc
from datachain.utils import suffix_to_number

if TYPE_CHECKING:
    from datachain.catalog.datasource import DataSource
    from datachain.client import Client
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.dataset import DatasetRecord
    from datachain.storage import Storage


class Listing:
    def __init__(
        self,
        storage: Optional["Storage"],
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        client: "Client",
        dataset: Optional["DatasetRecord"],
    ):
        self.storage = storage
        self.metastore = metastore
        self.warehouse = warehouse
        self.client = client
        self.dataset = dataset  # dataset representing bucket listing

    def clone(self) -> "Listing":
        return self.__class__(
            self.storage,
            self.metastore.clone(),
            self.warehouse.clone(),
            self.client,
            self.dataset,
        )

    def __enter__(self) -> "Listing":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.metastore.close()
        self.warehouse.close()

    @property
    def id(self):
        return self.storage.id

    @property
    def dataset_rows(self):
        return self.warehouse.dataset_rows(self.dataset, self.dataset.latest_version)

    def fetch(self, start_prefix="", method: str = "default") -> None:
        sync(get_loop(), self._fetch, start_prefix, method)

    async def _fetch(self, start_prefix: str, method: str) -> None:
        with self.clone() as fetch_listing:
            if start_prefix:
                start_prefix = start_prefix.rstrip("/")
            try:
                async for entries in fetch_listing.client.scandir(
                    start_prefix, method=method
                ):
                    fetch_listing.insert_entries(entries)
                    if len(entries) > 1:
                        fetch_listing.metastore.update_last_inserted_at()
            finally:
                fetch_listing.insert_entries_done()

    def insert_entry(self, entry: Entry) -> None:
        self.warehouse.insert_rows(
            self.dataset_rows.get_table(),
            self.warehouse.prepare_entries(self.client.uri, [entry]),
        )

    def insert_entries(self, entries: Iterable[Entry]) -> None:
        self.warehouse.insert_rows(
            self.dataset_rows.get_table(),
            self.warehouse.prepare_entries(self.client.uri, entries),
        )

    def insert_entries_done(self) -> None:
        self.warehouse.insert_rows_done(self.dataset_rows.get_table())

    def expand_path(self, path, use_glob=True) -> list[Node]:
        if use_glob and glob.has_magic(path):
            return self.warehouse.expand_path(self.dataset_rows, path)
        return [self.resolve_path(path)]

    def resolve_path(self, path) -> Node:
        return self.warehouse.get_node_by_path(self.dataset_rows, path)

    def ls_path(self, node, fields):
        if node.vtype == "tar" or node.dir_type == DirType.TAR_ARCHIVE:
            return self.warehouse.select_node_fields_by_parent_path_tar(
                self.dataset_rows, node.path, fields
            )
        return self.warehouse.select_node_fields_by_parent_path(
            self.dataset_rows, node.path, fields
        )

    def collect_nodes_to_instantiate(
        self,
        sources: Iterable["DataSource"],
        copy_to_filename: Optional[str],
        recursive=False,
        copy_dir_contents=False,
        relative_path=None,
        from_edatachain=False,
        from_dataset=False,
    ) -> list[NodeWithPath]:
        rel_path_elements = relative_path.split("/") if relative_path else []
        all_nodes: list[NodeWithPath] = []
        for src in sources:
            node = src.node
            if recursive and src.is_container():
                dir_path = []
                if not copy_dir_contents:
                    dir_path.append(node.name)
                subtree_nodes = src.find(sort=["path"])
                all_nodes.extend(
                    NodeWithPath(n.n, path=dir_path + n.path) for n in subtree_nodes
                )
            else:
                node_path = []
                if from_edatachain:
                    for rpe, npe in zip_longest(
                        rel_path_elements, node.path.split("/")
                    ):
                        if rpe == npe:
                            continue
                        if npe:
                            node_path.append(npe)
                elif copy_to_filename:
                    node_path = [os.path.basename(copy_to_filename)]
                elif from_dataset:
                    node_path = [
                        src.listing.client.name,
                        node.path,
                    ]
                else:
                    node_path = [node.name]
                all_nodes.append(NodeWithPath(node, path=node_path))
        return all_nodes

    def instantiate_nodes(
        self,
        all_nodes,
        output,
        total_files=None,
        force=False,
        shared_progress_bar=None,
    ):
        progress_bar = shared_progress_bar or tqdm(
            desc=f"Instantiating '{output}'",
            unit=" files",
            unit_scale=True,
            unit_divisor=1000,
            total=total_files,
        )

        counter = 0
        for node in all_nodes:
            dst = os.path.join(output, *node.path)
            dst_dir = os.path.dirname(dst)
            os.makedirs(dst_dir, exist_ok=True)
            uid = node.n.as_uid(self.client.uri)
            self.client.instantiate_object(uid, dst, progress_bar, force)
            counter += 1
            if counter > 1000:
                progress_bar.update(counter)
                counter = 0

        progress_bar.update(counter)

    def find(
        self,
        node,
        fields,
        names=None,
        inames=None,
        paths=None,
        ipaths=None,
        size=None,
        type=None,
        order_by=None,
    ):
        dr = self.dataset_rows
        conds = []
        if names:
            for name in names:
                conds.append(pathfunc.name(Column("path")).op("GLOB")(name))
        if inames:
            for iname in inames:
                conds.append(
                    func.lower(pathfunc.name(Column("path"))).op("GLOB")(iname.lower())
                )
        if paths:
            for path in paths:
                conds.append(Column("path").op("GLOB")(path))
        if ipaths:
            for ipath in ipaths:
                conds.append(func.lower(Column("path")).op("GLOB")(ipath.lower()))

        if size is not None:
            size_limit = suffix_to_number(size)
            if size_limit >= 0:
                conds.append(Column("size") >= size_limit)
            else:
                conds.append(Column("size") <= -size_limit)

        return self.warehouse.find(
            dr,
            node,
            fields,
            type=type,
            conds=conds,
            order_by=order_by,
        )

    def du(self, node: Node, count_files: bool = False):
        return self.warehouse.size(self.dataset_rows, node, count_files)

    def subtree_files(self, node: Node, sort=None):
        if node.dir_type == DirType.TAR_ARCHIVE or node.vtype != "":
            include_subobjects = True
        else:
            include_subobjects = False

        return self.warehouse.get_subtree_files(
            self.dataset_rows,
            node,
            sort=sort,
            include_subobjects=include_subobjects,
        )

    def get_dirs_by_parent_path(
        self,
        parent_path: str,
    ) -> Iterator[Node]:
        return self.warehouse.get_dirs_by_parent_path(self.dataset_rows, parent_path)
