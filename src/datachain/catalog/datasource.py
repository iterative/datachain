from collections.abc import Iterable

from datachain.node import DirType, NodeWithPath


class DataSource:
    def __init__(self, listing, node, as_container=False):
        self.listing = listing
        self.node = node
        self.as_container = (
            as_container  # Indicates whether a .tar file is handled as a container
        )

    def get_full_path(self):
        return self.get_node_full_path(self.node)

    def get_node_full_path(self, node):
        return self.listing.client.get_full_path(node.full_path)

    def get_node_full_path_from_path(self, full_path):
        return self.listing.client.get_full_path(full_path)

    def is_single_object(self):
        return self.node.dir_type == DirType.FILE or (
            not self.as_container and self.node.dir_type == DirType.TAR_ARCHIVE
        )

    def is_container(self):
        return not self.is_single_object()

    def ls(self, fields) -> Iterable[tuple]:
        if self.is_single_object():
            return [tuple(getattr(self.node, f) for f in fields)]
        return self.listing.ls_path(self.node, fields)

    def dirname(self):
        if self.is_single_object():
            return self.node.parent
        return self.node.path

    def find(self, *, sort=None):
        if self.is_single_object():
            return [NodeWithPath(self.node, [])]

        return self.listing.subtree_files(self.node, sort=sort)
