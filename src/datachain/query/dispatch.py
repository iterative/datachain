import contextlib
from collections.abc import Iterable, Sequence
from itertools import chain
from multiprocessing import cpu_count
from sys import stdin
from threading import Timer
from typing import TYPE_CHECKING, Literal, Optional

import multiprocess
from cloudpickle import load, loads
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from multiprocess import get_context

from datachain.catalog import Catalog
from datachain.catalog.catalog import clone_catalog_with_cache
from datachain.catalog.loader import DISTRIBUTED_IMPORT_PATH, get_udf_distributor_class
from datachain.lib.udf import _get_cache
from datachain.query.batch import RowsOutput, RowsOutputBatch
from datachain.query.dataset import (
    get_download_callback,
    get_generated_callback,
    get_processed_callback,
    process_udf_outputs,
)
from datachain.query.queue import get_from_queue, put_into_queue
from datachain.query.udf import UdfInfo
from datachain.query.utils import get_query_id_column
from datachain.utils import batched, flatten, safe_closing

if TYPE_CHECKING:
    from sqlalchemy import Select, Table

    from datachain.data_storage import AbstractMetastore, AbstractWarehouse
    from datachain.lib.udf import UDFAdapter

DEFAULT_BATCH_SIZE = 10000
STOP_SIGNAL = "STOP"
OK_STATUS = "OK"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"
NOTIFY_STATUS = "NOTIFY"


def get_n_workers_from_arg(n_workers: Optional[int] = None) -> int:
    if not n_workers:
        return cpu_count()
    if n_workers < 1:
        raise RuntimeError("Must use at least one worker for parallel UDF execution!")
    return n_workers


def udf_entrypoint() -> int:
    # Load UDF info from stdin
    udf_info: UdfInfo = load(stdin.buffer)

    # Parallel processing (faster for more CPU-heavy UDFs)
    dispatch = UDFDispatcher(udf_info)

    query = udf_info["query"]
    rows_total = udf_info["rows_total"]
    batching = udf_info["batching"]
    is_generator = udf_info["is_generator"]
    n_workers = udf_info["processes"]
    if n_workers is True:
        n_workers = None  # Use default number of CPUs (cores)

    wh_cls, wh_args, wh_kwargs = udf_info["warehouse_clone_params"]
    warehouse: AbstractWarehouse = wh_cls(*wh_args, **wh_kwargs)

    with contextlib.closing(
        batching(warehouse.dataset_select_paginated, query, ids_only=True)
    ) as udf_inputs:
        download_cb = get_download_callback()
        processed_cb = get_processed_callback()
        generated_cb = get_generated_callback(is_generator)
        try:
            dispatch.run_udf_parallel(
                udf_inputs,
                rows_total=rows_total,
                n_workers=n_workers,
                download_cb=download_cb,
                processed_cb=processed_cb,
                generated_cb=generated_cb,
            )
        finally:
            download_cb.close()
            processed_cb.close()
            generated_cb.close()

    return 0


def udf_worker_entrypoint() -> int:
    if not (udf_distributor_class := get_udf_distributor_class()):
        raise RuntimeError(
            f"{DISTRIBUTED_IMPORT_PATH} import path is required "
            "for distributed UDF processing."
        )
    return udf_distributor_class.run_worker()


class UDFDispatcher:
    catalog: Optional[Catalog] = None
    task_queue: Optional[multiprocess.Queue] = None
    done_queue: Optional[multiprocess.Queue] = None

    def __init__(self, udf_info: UdfInfo, buffer_size: int = DEFAULT_BATCH_SIZE):
        self.udf_data = udf_info["udf_data"]
        self.catalog_init_params = udf_info["catalog_init"]
        self.metastore_clone_params = udf_info["metastore_clone_params"]
        self.warehouse_clone_params = udf_info["warehouse_clone_params"]
        self.query = udf_info["query"]
        self.table = udf_info["table"]
        self.udf_fields = udf_info["udf_fields"]
        self.cache = udf_info["cache"]
        self.is_generator = udf_info["is_generator"]
        self.is_batching = udf_info["batching"].is_batching
        self.buffer_size = buffer_size
        self.catalog = None
        self.task_queue = None
        self.done_queue = None
        self.ctx = get_context("spawn")

    def _create_worker(self) -> "UDFWorker":
        if not self.catalog:
            ms_cls, ms_args, ms_kwargs = self.metastore_clone_params
            metastore: AbstractMetastore = ms_cls(*ms_args, **ms_kwargs)
            ws_cls, ws_args, ws_kwargs = self.warehouse_clone_params
            warehouse: AbstractWarehouse = ws_cls(*ws_args, **ws_kwargs)
            self.catalog = Catalog(metastore, warehouse, **self.catalog_init_params)
        self.udf = loads(self.udf_data)
        return UDFWorker(
            self.catalog,
            self.udf,
            self.task_queue,
            self.done_queue,
            self.query,
            self.table,
            self.is_batching,
            self.cache,
            self.udf_fields,
        )

    def _run_worker(self) -> None:
        try:
            worker = self._create_worker()
            worker.run()
        except (Exception, KeyboardInterrupt) as e:
            if self.done_queue:
                put_into_queue(
                    self.done_queue,
                    {"status": FAILED_STATUS, "exception": e},
                )
            raise

    @staticmethod
    def send_stop_signal_to_workers(task_queue, n_workers: Optional[int] = None):
        n_workers = get_n_workers_from_arg(n_workers)
        for _ in range(n_workers):
            put_into_queue(task_queue, STOP_SIGNAL)

    def run_udf_parallel(  # noqa: C901, PLR0912
        self,
        input_rows: Iterable[RowsOutput],
        rows_total: int,
        n_workers: Optional[int] = None,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
        generated_cb: Callback = DEFAULT_CALLBACK,
    ) -> None:
        n_workers = get_n_workers_from_arg(n_workers)

        input_batch_size = rows_total // n_workers
        if input_batch_size == 0:
            input_batch_size = 1
        elif input_batch_size > DEFAULT_BATCH_SIZE:
            input_batch_size = DEFAULT_BATCH_SIZE

        if self.buffer_size < n_workers:
            raise RuntimeError(
                "Parallel run error: buffer size is smaller than "
                f"number of workers: {self.buffer_size} < {n_workers}"
            )

        self.task_queue = self.ctx.Queue()
        self.done_queue = self.ctx.Queue()
        pool = [
            self.ctx.Process(name=f"Worker-UDF-{i}", target=self._run_worker)
            for i in range(n_workers)
        ]
        for p in pool:
            p.start()

        # Will be set to True if all tasks complete normally
        normal_completion = False
        try:
            # Will be set to True when the input is exhausted
            input_finished = False

            if not self.is_batching:
                input_rows = batched(flatten(input_rows), input_batch_size)

            # Stop all workers after the input rows have finished processing
            input_data = chain(input_rows, [STOP_SIGNAL] * n_workers)

            # Add initial buffer of tasks
            for _ in range(self.buffer_size):
                try:
                    put_into_queue(self.task_queue, next(input_data))
                except StopIteration:
                    input_finished = True
                    break

            # Process all tasks
            while n_workers > 0:
                result = get_from_queue(self.done_queue)

                if downloaded := result.get("downloaded"):
                    download_cb.relative_update(downloaded)
                if processed := result.get("processed"):
                    processed_cb.relative_update(processed)
                if generated := result.get("generated"):
                    generated_cb.relative_update(generated)

                status = result["status"]
                if status in (OK_STATUS, NOTIFY_STATUS):
                    pass  # Do nothing here
                elif status == FINISHED_STATUS:
                    n_workers -= 1  # Worker finished
                else:  # Failed / error
                    n_workers -= 1
                    if exc := result.get("exception"):
                        raise exc
                    raise RuntimeError("Internal error: Parallel UDF execution failed")

                if status == OK_STATUS and not input_finished:
                    try:
                        put_into_queue(self.task_queue, next(input_data))
                    except StopIteration:
                        input_finished = True

            # Finished with all tasks normally
            normal_completion = True
        finally:
            if not normal_completion:
                # Stop all workers if there is an unexpected exception
                for _ in pool:
                    put_into_queue(self.task_queue, STOP_SIGNAL)
                self.task_queue.close()

                # This allows workers (and this process) to exit without
                # consuming any remaining data in the queues.
                # (If they exit due to an exception.)
                self.task_queue.cancel_join_thread()
                self.done_queue.cancel_join_thread()

                # Flush all items from the done queue.
                # This is needed if any workers are still running.
                while n_workers > 0:
                    result = get_from_queue(self.done_queue)
                    status = result["status"]
                    if status != OK_STATUS:
                        n_workers -= 1

            # Wait for workers to stop
            for p in pool:
                p.join()


class DownloadCallback(Callback):
    def __init__(self, queue: "multiprocess.Queue") -> None:
        self.queue = queue
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        # This callback is used to notify the size of the downloaded files
        pass

    def increment_file_count(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, "downloaded": inc})


class ProcessedCallback(Callback):
    def __init__(
        self,
        name: Literal["processed", "generated"],
        queue: "multiprocess.Queue",
    ) -> None:
        self.name = name
        self.queue = queue
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, self.name: inc})


class UDFWorker:
    def __init__(
        self,
        catalog: "Catalog",
        udf: "UDFAdapter",
        task_queue: "multiprocess.Queue",
        done_queue: "multiprocess.Queue",
        query: "Select",
        table: "Table",
        is_batching: bool,
        cache: bool,
        udf_fields: Sequence[str],
    ) -> None:
        self.catalog = catalog
        self.udf = udf
        self.task_queue = task_queue
        self.done_queue = done_queue
        self.query = query
        self.table = table
        self.is_batching = is_batching
        self.cache = cache
        self.udf_fields = udf_fields

        self.download_cb = DownloadCallback(self.done_queue)
        self.processed_cb = ProcessedCallback("processed", self.done_queue)
        self.generated_cb = ProcessedCallback("generated", self.done_queue)

    def run(self) -> None:
        prefetch = self.udf.prefetch
        with _get_cache(self.catalog.cache, prefetch, use_cache=self.cache) as _cache:
            catalog = clone_catalog_with_cache(self.catalog, _cache)
            udf_results = self.udf.run(
                self.udf_fields,
                self.get_inputs(),
                catalog,
                self.cache,
                download_cb=self.download_cb,
                processed_cb=self.processed_cb,
            )
            with safe_closing(udf_results):
                process_udf_outputs(
                    catalog.warehouse,
                    self.table,
                    self.notify_and_process(udf_results),
                    self.udf,
                    cb=self.generated_cb,
                )
        put_into_queue(self.done_queue, {"status": FINISHED_STATUS})

    def notify_and_process(self, udf_results):
        for row in udf_results:
            put_into_queue(self.done_queue, {"status": OK_STATUS})
            yield row

    def get_inputs(self):
        warehouse = self.catalog.warehouse.clone()
        col_id = get_query_id_column(self.query)

        if self.is_batching:
            while (batch := get_from_queue(self.task_queue)) != STOP_SIGNAL:
                ids = [row[0] for row in batch.rows]
                rows = warehouse.dataset_rows_select(self.query.where(col_id.in_(ids)))
                yield RowsOutputBatch(list(rows))
        else:
            while (batch := get_from_queue(self.task_queue)) != STOP_SIGNAL:
                yield from warehouse.dataset_rows_select(
                    self.query.where(col_id.in_(batch))
                )


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)
