from fsspec import Callback
from fsspec.callbacks import TqdmCallback


class CombinedDownloadCallback(Callback):
    def set_size(self, size):
        # This is a no-op to prevent fsspec's .get_file() from setting the combined
        # download size to the size of the current file.
        pass

    def increment_file_count(self, n: int = 1) -> None:
        pass


class TqdmCombinedDownloadCallback(CombinedDownloadCallback, TqdmCallback):
    def __init__(self, tqdm_kwargs=None, *args, **kwargs):
        self.files_count = 0
        super().__init__(tqdm_kwargs, *args, **kwargs)

    def increment_file_count(self, n: int = 1) -> None:
        self.files_count += n
        if self.tqdm is not None:
            self.tqdm.postfix = f"{self.files_count} files"
