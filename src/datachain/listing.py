import glob
import os
from collections.abc import Iterable, Iterator
from functools import cached_property
from itertools import zip_longest
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Column
from sqlalchemy.sql import func
from tqdm import tqdm

from datachain.node import DirType, Node, NodeWithPath
from datachain.sql.functions import path as pathfunc
from datachain.utils import suffix_to_number

if TYPE_CHECKING:
    from datachain.catalog.datasource import DataSource
    from datachain.client import Client
    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.dataset import DatasetRecord


class Listing:
    def __init__(
        self,
        metastore: "AbstractMetastore",
        warehouse: "AbstractWarehouse",
        client: "Client",
        dataset_name: Optional["str"] = None,
        dataset_version: Optional[int] = None,
        object_name: str = "file",
    ):
        self.metastore = metastore
        self.warehouse = warehouse
        self.client = client
        self.dataset_name = dataset_name  # dataset representing bucket listing
        self.dataset_version = dataset_version  # dataset representing bucket listing
        self.object_name = object_name

    def clone(self) -> "Listing":
        return self.__class__(
            self.metastore.clone(),
            self.warehouse.clone(),
            self.client,
            self.dataset_name,
            self.dataset_version,
            self.object_name,
        )

    def __enter__(self) -> "Listing":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.warehouse.close()

    @property
    def uri(self):
        from datachain.lib.listing import listing_uri_from_name

        assert self.dataset_name

        return listing_uri_from_name(self.dataset_name)

    @cached_property
    def dataset(self) -> "DatasetRecord":
        assert self.dataset_name
        return self.metastore.get_dataset(self.dataset_name)

    @cached_property
    def dataset_rows(self):
        dataset = self.dataset
        return self.warehouse.dataset_rows(
            dataset,
            self.dataset_version or dataset.latest_version,
            object_name=self.object_name,
        )

    def expand_path(self, path, use_glob=True) -> list[Node]:
        if use_glob and glob.has_magic(path):
            return self.warehouse.expand_path(self.dataset_rows, path)
        return [self.resolve_path(path)]

    def resolve_path(self, path) -> Node:
        return self.warehouse.get_node_by_path(self.dataset_rows, path)

    def ls_path(self, node, fields):
        if node.location or node.dir_type == DirType.TAR_ARCHIVE:
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
        all_nodes: Iterable[NodeWithPath],
        output,
        total_files=None,
        force=False,
        shared_progress_bar=None,
    ) -> None:
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
            file = node.n.to_file(self.client.uri)
            self.client.instantiate_object(file, dst, progress_bar, force)
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
                conds.append(
                    pathfunc.name(Column(dr.col_name("path"))).op("GLOB")(name)
                )
        if inames:
            for iname in inames:
                conds.append(
                    func.lower(pathfunc.name(Column(dr.col_name("path")))).op("GLOB")(
                        iname.lower()
                    )
                )
        if paths:
            for path in paths:
                conds.append(Column(dr.col_name("path")).op("GLOB")(path))
        if ipaths:
            for ipath in ipaths:
                conds.append(
                    func.lower(Column(dr.col_name("path"))).op("GLOB")(ipath.lower())
                )

        if size is not None:
            size_limit = suffix_to_number(size)
            if size_limit >= 0:
                conds.append(Column(dr.col_name("size")) >= size_limit)
            else:
                conds.append(Column(dr.col_name("size")) <= -size_limit)

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
        if node.dir_type == DirType.TAR_ARCHIVE or node.location:
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
