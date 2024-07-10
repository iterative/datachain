from datachain.lib.utils import DataChainParamsError


class SettingsError(DataChainParamsError):
    def __init__(self, msg):
        super().__init__(f"Dataset settings error: {msg}")


class Settings:
    def __init__(
        self, cache=None, batch=None, parallel=None, workers=None, min_task_size=None
    ):
        self._cache = cache
        self._batch = batch
        self.parallel = parallel
        self._workers = workers
        self.min_task_size = min_task_size

        if not isinstance(cache, bool) and cache is not None:
            raise SettingsError(
                "'cache' argument must be bool"
                f" while {cache.__class__.__name__} was given"
            )

        if not isinstance(batch, int) and batch is not None:
            raise SettingsError(
                "'batch' argument must be int or None"
                f" while {batch.__class__.__name__} was given"
            )

        if not isinstance(parallel, int) and parallel is not None:
            raise SettingsError(
                "'parallel' argument must be int or None"
                f" while {parallel.__class__.__name__} was given"
            )

        if (
            not isinstance(workers, bool)
            and not isinstance(workers, int)
            and workers is not None
        ):
            raise SettingsError(
                "'workers' argument must be int or bool"
                f" while {workers.__class__.__name__} was given"
            )

        if min_task_size is not None and not isinstance(min_task_size, int):
            raise SettingsError(
                "'min_task_size' argument must be int or None"
                f", {min_task_size.__class__.__name__} was given"
            )

    @property
    def cache(self):
        return self._cache if self._cache is not None else False

    @property
    def batch(self):
        return self._batch if self._batch is not None else 1

    @property
    def workers(self):
        return self._workers if self._workers is not None else False

    def to_dict(self):
        res = {}
        if self._cache is not None:
            res["cache"] = self.cache
        if self._batch is not None:
            res["batch"] = self.batch
        if self.parallel is not None:
            res["parallel"] = self.parallel
        if self._workers is not None:
            res["workers"] = self.workers
        if self.min_task_size is not None:
            res["min_task_size"] = self.min_task_size
        return res

    def add(self, settings: "Settings"):
        self._cache = settings._cache or self._cache
        self._batch = settings._batch or self._batch
        self.parallel = settings.parallel or self.parallel
        self._workers = settings._workers or self._workers
        self.min_task_size = settings.min_task_size or self.min_task_size
