import logging

from datachain.nodes_thread_pool import NodesThreadPool

logger = logging.getLogger("datachain")


class NodesFetcher(NodesThreadPool):
    def __init__(self, client, max_threads, cache):
        super().__init__(max_threads)
        self.client = client
        self.cache = cache
        logger.info("NodesFetcher initialized with max_threads: %d", max_threads)

    def done_task(self, done):
        for task in done:
            try:
                task.result()
            except Exception as e:
                logger.error("Task resulted in an error: %s", e)

    def do_task(self, chunk):
        from fsspec import Callback

        class _CB(Callback):
            def relative_update(_, inc: int = 1):  # noqa: N805
                self.increase_counter(inc)

        for node in chunk:
            uid = node.as_uid(self.client.uri)
            if self.cache.contains(uid):
                logger.debug("Cache hit for UID: %s", uid)
                self.increase_counter(node.size)
            else:
                logger.debug("Cache miss for UID: %s, adding to cache", uid)
                self.client.put_in_cache(uid, callback=_CB())
