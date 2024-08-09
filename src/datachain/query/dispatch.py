import contextlib
from collections.abc import Iterator, Sequence
from itertools import chain
from multiprocessing import cpu_count
from sys import stdin
from typing import Optional

import attrs
import multiprocess
from cloudpickle import load, loads
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from multiprocess import get_context

from datachain.catalog import Catalog
from datachain.catalog.loader import get_distributed_class
from datachain.query.dataset import (
    get_download_callback,
    get_generated_callback,
    get_processed_callback,
    process_udf_outputs,
)
from datachain.query.queue import (
    get_from_queue,
    marshal,
    msgpack_pack,
    msgpack_unpack,
    put_into_queue,
    unmarshal,
)
from datachain.query.udf import UDFBase, UDFFactory, UDFResult
from datachain.utils import batched_it

DEFAULT_BATCH_SIZE = 10000
STOP_SIGNAL = "STOP"
OK_STATUS = "OK"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"
NOTIFY_STATUS = "NOTIFY"


def full_module_type_path(typ: type) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


def get_n_workers_from_arg(n_workers: Optional[int] = None) -> int:
    if not n_workers:
        return cpu_count()
    if n_workers < 1:
        raise RuntimeError("Must use at least one worker for parallel UDF execution!")
    return n_workers


def udf_entrypoint() -> int:
    # Load UDF info from stdin
    udf_info = load(stdin.buffer)

    (
        warehouse_class,
        warehouse_args,
        warehouse_kwargs,
    ) = udf_info["warehouse_clone_params"]
    warehouse = warehouse_class(*warehouse_args, **warehouse_kwargs)

    # Parallel processing (faster for more CPU-heavy UDFs)
    dispatch = UDFDispatcher(
        udf_info["udf_data"],
        udf_info["catalog_init"],
        udf_info["id_generator_clone_params"],
        udf_info["metastore_clone_params"],
        udf_info["warehouse_clone_params"],
        udf_fields=udf_info["udf_fields"],
        cache=udf_info["cache"],
        is_generator=udf_info.get("is_generator", False),
    )

    query = udf_info["query"]
    batching = udf_info["batching"]
    table = udf_info["table"]
    n_workers = udf_info["processes"]
    udf = loads(udf_info["udf_data"])
    if n_workers is True:
        # Use default number of CPUs (cores)
        n_workers = None

    with contextlib.closing(
        batching(warehouse.dataset_select_paginated, query)
    ) as udf_inputs:
        download_cb = get_download_callback()
        processed_cb = get_processed_callback()
        generated_cb = get_generated_callback(dispatch.is_generator)
        try:
            udf_results = dispatch.run_udf_parallel(
                marshal(udf_inputs),
                n_workers=n_workers,
                processed_cb=processed_cb,
                download_cb=download_cb,
            )
            process_udf_outputs(warehouse, table, udf_results, udf, cb=generated_cb)
        finally:
            download_cb.close()
            processed_cb.close()
            generated_cb.close()

    warehouse.insert_rows_done(table)

    return 0


def udf_worker_entrypoint() -> int:
    return get_distributed_class().run_worker()


class UDFDispatcher:
    catalog: Optional[Catalog] = None
    task_queue: Optional[multiprocess.Queue] = None
    done_queue: Optional[multiprocess.Queue] = None
    _batch_size: Optional[int] = None

    def __init__(
        self,
        udf_data,
        catalog_init_params,
        id_generator_clone_params,
        metastore_clone_params,
        warehouse_clone_params,
        udf_fields: "Sequence[str]",
        cache: bool,
        is_generator: bool = False,
        buffer_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.udf_data = udf_data
        self.catalog_init_params = catalog_init_params
        (
            self.id_generator_class,
            self.id_generator_args,
            self.id_generator_kwargs,
        ) = id_generator_clone_params
        (
            self.metastore_class,
            self.metastore_args,
            self.metastore_kwargs,
        ) = metastore_clone_params
        (
            self.warehouse_class,
            self.warehouse_args,
            self.warehouse_kwargs,
        ) = warehouse_clone_params
        self.udf_fields = udf_fields
        self.cache = cache
        self.is_generator = is_generator
        self.buffer_size = buffer_size
        self.catalog = None
        self.task_queue = None
        self.done_queue = None
        self.ctx = get_context("spawn")

    @property
    def batch_size(self):
        if not self.udf:
            self.udf = self.udf_factory()
        if self._batch_size is None:
            if hasattr(self.udf, "properties") and hasattr(
                self.udf.properties, "batch"
            ):
                self._batch_size = self.udf.properties.batch
            else:
                self._batch_size = 1
        return self._batch_size

    def _create_worker(self) -> "UDFWorker":
        if not self.catalog:
            id_generator = self.id_generator_class(
                *self.id_generator_args, **self.id_generator_kwargs
            )
            metastore = self.metastore_class(
                *self.metastore_args, **self.metastore_kwargs
            )
            warehouse = self.warehouse_class(
                *self.warehouse_args, **self.warehouse_kwargs
            )
            self.catalog = Catalog(
                id_generator, metastore, warehouse, **self.catalog_init_params
            )
        udf = loads(self.udf_data)
        # isinstance cannot be used here, as cloudpickle packages the entire class
        # definition, and so these two types are not considered exactly equal,
        # even if they have the same import path.
        if full_module_type_path(type(udf)) != full_module_type_path(UDFFactory):
            self.udf = udf
        else:
            self.udf = None
            self.udf_factory = udf
        if not self.udf:
            self.udf = self.udf_factory()

        return UDFWorker(
            self.catalog,
            self.udf,
            self.task_queue,
            self.done_queue,
            self.is_generator,
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

    def create_input_queue(self):
        return self.ctx.Queue()

    def run_udf_parallel(  # noqa: C901, PLR0912
        self,
        input_rows,
        n_workers: Optional[int] = None,
        input_queue=None,
        processed_cb: Callback = DEFAULT_CALLBACK,
        download_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Sequence[UDFResult]]:
        n_workers = get_n_workers_from_arg(n_workers)

        if self.buffer_size < n_workers:
            raise RuntimeError(
                "Parallel run error: buffer size is smaller than "
                f"number of workers: {self.buffer_size} < {n_workers}"
            )

        if input_queue:
            streaming_mode = True
            self.task_queue = input_queue
        else:
            streaming_mode = False
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

            if not streaming_mode:
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
                status = result["status"]
                if status == NOTIFY_STATUS:
                    if downloaded := result.get("downloaded"):
                        download_cb.relative_update(downloaded)
                    if processed := result.get("processed"):
                        processed_cb.relative_update(processed)
                elif status == FINISHED_STATUS:
                    # Worker finished
                    n_workers -= 1
                elif status == OK_STATUS:
                    if processed := result.get("processed"):
                        processed_cb.relative_update(processed)
                    yield msgpack_unpack(result["result"])
                else:  # Failed / error
                    n_workers -= 1
                    if exc := result.get("exception"):
                        raise exc
                    raise RuntimeError("Internal error: Parallel UDF execution failed")

                if status == OK_STATUS and not streaming_mode and not input_finished:
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


class WorkerCallback(Callback):
    def __init__(self, queue: "multiprocess.Queue"):
        self.queue = queue
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        put_into_queue(self.queue, {"status": NOTIFY_STATUS, "downloaded": inc})


class ProcessedCallback(Callback):
    def __init__(self):
        self.processed_rows: Optional[int] = None
        super().__init__()

    def relative_update(self, inc: int = 1) -> None:
        self.processed_rows = inc


@attrs.define
class UDFWorker:
    catalog: Catalog
    udf: UDFBase
    task_queue: "multiprocess.Queue"
    done_queue: "multiprocess.Queue"
    is_generator: bool
    cache: bool
    udf_fields: Sequence[str]
    cb: Callback = attrs.field()

    @cb.default
    def _default_callback(self) -> WorkerCallback:
        return WorkerCallback(self.done_queue)

    def run(self) -> None:
        processed_cb = ProcessedCallback()
        udf_results = self.udf.run(
            self.udf_fields,
            unmarshal(self.get_inputs()),
            self.catalog,
            self.is_generator,
            self.cache,
            download_cb=self.cb,
            processed_cb=processed_cb,
        )
        for udf_output in udf_results:
            for batch in batched_it(udf_output, DEFAULT_BATCH_SIZE):
                put_into_queue(
                    self.done_queue,
                    {
                        "status": OK_STATUS,
                        "result": msgpack_pack(list(batch)),
                    },
                )
            put_into_queue(
                self.done_queue,
                {"status": NOTIFY_STATUS, "processed": processed_cb.processed_rows},
            )
        put_into_queue(self.done_queue, {"status": FINISHED_STATUS})

    def get_inputs(self):
        while (batch := get_from_queue(self.task_queue)) != STOP_SIGNAL:
            yield batch
