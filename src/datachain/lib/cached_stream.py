from abc import ABC
from contextlib import AbstractContextManager

from datachain.cache import UniqueId


class AbstractCachedStream(AbstractContextManager, ABC):
    def __init__(self, catalog, uid: UniqueId):
        self.catalog = catalog
        self.uid = uid
        self.mode = "rb"

    def set_mode(self, mode):
        self.mode = mode


class PreCachedStream(AbstractCachedStream):
    def __init__(self, catalog, uid: UniqueId):
        super().__init__(catalog, uid)
        self.client = self.catalog.get_client(self.uid.storage)
        self.cached_file = None

    def get_path_in_cache(self):
        return self.catalog.cache.path_from_checksum(self.uid.get_hash())

    def __enter__(self):
        self.client.download(self.uid)
        self.cached_file = open(self.get_path_in_cache(), self.mode)
        return self.cached_file

    def __exit__(self, *args):
        self.cached_file.close()


class PreDownloadStream(PreCachedStream):
    def __exit__(self, *args):
        super().__exit__(*args)
        self.catalog.cache.remove(self.uid)
