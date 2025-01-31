import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from datachain.nodes_thread_pool import NodesThreadPool

if TYPE_CHECKING:
    from datachain.cache import Cache
    from datachain.client.fsspec import Client
    from datachain.node import Node

logger = logging.getLogger("datachain")


class NodesFetcher(NodesThreadPool):
    def __init__(self, client: "Client", max_threads: int, cache: "Cache"):
        super().__init__(max_threads)
        self.client = client
        self.cache = cache

    def done_task(self, done):
        for task in done:
            task.result()

    def do_task(self, chunk: Iterable["Node"]) -> None:
        from fsspec import Callback

        class _CB(Callback):
            def relative_update(_, inc: int = 1):  # noqa: N805
                self.increase_counter(inc)

        for node in chunk:
            file = node.to_file(self.client.uri)
            if self.cache.contains(file):
                self.increase_counter(node.size)
            else:
                self.client.put_in_cache(file, callback=_CB())
