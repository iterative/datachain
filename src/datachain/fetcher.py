import logging

from datachain.runner_thread_pool import RunnerThreadPool

logger = logging.getLogger("datachain")


class NodesFetcher(RunnerThreadPool):
    def __init__(self, client, max_threads, cache):
        super().__init__(max_threads)
        self.client = client
        self.cache = cache

    def done_task(self, done):
        for task in done:
            task.result()

    def do_task(self, chunk):
        from fsspec import Callback

        class _CB(Callback):
            def relative_update(_, inc: int = 1):  # noqa: N805
                self.increase_counter(inc)

        for node in chunk:
            uid = node.as_uid(self.client.uri)
            if self.cache.contains(uid):
                self.increase_counter(node.size)
            else:
                self.client.put_in_cache(uid, callback=_CB())


class FileFetcher(RunnerThreadPool):
    def __init__(self, catalog, max_threads, cache):
        super().__init__(max_threads)
        self.catalog = catalog
        self.cache = cache
        self.clients = {}

    def done_task(self, done):
        for task in done:
            task.result()

    def get_client(self, source: str):
        if source not in self.clients:
            self.clients[source] = self.catalog.get_client(source)
        return self.clients[source]

    def do_task(self, chunk):
        from fsspec import Callback

        class _CB(Callback):
            def relative_update(_, inc: int = 1):  # noqa: N805
                self.increase_counter(inc)

        for file in chunk:
            uid = file.get_uid()
            if self.cache.contains(uid):
                # self.increase_counter(file.size)
                self.increase_counter(1)
            else:
                self.get_client(file.source).put_in_cache(uid, callback=_CB())
