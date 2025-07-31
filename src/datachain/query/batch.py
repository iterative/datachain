import contextlib
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from typing import Callable, Optional, Union

import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID
from datachain.query.utils import get_query_column

RowsOutputBatch = Sequence[Sequence]
RowsOutput = Union[Sequence, RowsOutputBatch]

OBJECT_OVERHEAD_BYTES = 100


class BatchingStrategy(ABC):
    """BatchingStrategy provides means of batching UDF executions."""

    is_batching: bool

    @abstractmethod
    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: Optional[sa.ColumnElement] = None,
    ) -> Generator[RowsOutput, None, None]:
        """Apply the provided parameters to the UDF."""


class NoBatching(BatchingStrategy):
    """
    NoBatching implements the default batching strategy, which is not to
    batch UDF calls.
    """

    is_batching = False

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: Optional[sa.ColumnElement] = None,
    ) -> Generator[Sequence, None, None]:
        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col)
            ids_only = True

        rows = execute(query)
        yield from (r[0] for r in rows) if ids_only else rows


class Batch(BatchingStrategy):
    """
    Batch implements UDF call batching, where each execution of a UDF
    is passed a sequence of multiple parameter sets.
    """

    is_batching = True

    def __init__(self, count: int):
        self.count = count

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: Optional[sa.ColumnElement] = None,
    ) -> Generator[RowsOutput, None, None]:
        from datachain.data_storage.warehouse import SELECT_BATCH_SIZE

        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col)
            ids_only = True

        # choose page size that is a multiple of the batch size
        page_size = math.ceil(SELECT_BATCH_SIZE / self.count) * self.count

        # select rows in batches
        results = []

        with contextlib.closing(execute(query, page_size=page_size)) as batch_rows:
            for row in batch_rows:
                results.append(row)
                if len(results) >= self.count:
                    batch, results = results[: self.count], results[self.count :]
                    yield [r[0] for r in batch] if ids_only else batch

            if len(results) > 0:
                yield [r[0] for r in results] if ids_only else results


class DynamicBatch(BatchingStrategy):
    """
    DynamicBatch implements UDF call batching with dynamic batch sizes based on
    both row count and memory usage limits.
    """

    is_batching = True

    def __init__(
        self,
        max_rows: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        is_input_batched: bool = True,
    ):
        self.max_rows = max_rows or 2000
        self.max_memory_mb = max_memory_mb or 1000
        self.max_memory_bytes = self.max_memory_mb * 1024 * 1024
        self.is_input_batched = is_input_batched
        # If we yield individual rows, set is_batching to False
        self.is_batching = is_input_batched

    def _estimate_row_memory(self, row) -> int:
        """Estimate memory usage of a row in bytes."""
        if not row:
            return 0

        total_size = 0
        for item in row:
            if isinstance(item, (str, bytes, int, float, bool)):
                total_size += sys.getsizeof(item)
            elif isinstance(item, (list, tuple)):
                total_size += sys.getsizeof(item)
                for subitem in item:
                    total_size += sys.getsizeof(subitem)
            else:
                # For complex objects, use a conservative estimate
                total_size += (
                    sys.getsizeof(item) + OBJECT_OVERHEAD_BYTES
                )  # Add buffer for object overhead

        return total_size

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: Optional[sa.ColumnElement] = None,
    ) -> Generator[RowsOutput, None, None]:
        import psutil

        from datachain.data_storage.warehouse import SELECT_BATCH_SIZE

        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col)
            ids_only = True

        # Use a larger page size for efficiency, but we'll batch dynamically
        page_size = max(SELECT_BATCH_SIZE, self.max_rows * 2)

        # select rows in batches
        results: list[Sequence] = []
        current_memory = 0
        row_count = 0

        with contextlib.closing(execute(query, page_size=page_size)) as chunk_rows:
            for row in chunk_rows:
                row_memory = self._estimate_row_memory(row)
                row_count += 1

                # Check if adding this row would exceed limits
                # Also check system memory usage every 10 rows
                # (same as in process_udf_outputs)
                should_yield = (
                    len(results) >= self.max_rows
                    or current_memory + row_memory > self.max_memory_bytes
                    or (row_count % 100 == 0 and psutil.virtual_memory().percent > 80)
                )

                if should_yield and results:  # Yield current batch if we have one
                    if self.is_input_batched:
                        # Yield the entire batch
                        yield [r[0] for r in results] if ids_only else results
                    else:
                        # Yield individual rows
                        for result in results:
                            yield [result[0]] if ids_only else result
                    results = []
                    current_memory = 0

                results.append(row)
                current_memory += row_memory

            if len(results) > 0:
                if self.is_input_batched:
                    # Yield the entire batch
                    yield [r[0] for r in results] if ids_only else results
                else:
                    # Yield individual rows
                    for result in results:
                        yield [result[0]] if ids_only else result


class Partition(BatchingStrategy):
    """
    Partition implements UDF call batching, where each execution of a UDF
    is run on a list of dataset rows grouped by the specified column.
    Dataset rows need to be sorted by the grouping column.
    """

    is_batching = True

    def __call__(
        self,
        execute: Callable,
        query: sa.Select,
        id_col: Optional[sa.ColumnElement] = None,
    ) -> Generator[RowsOutput, None, None]:
        if (partition_col := get_query_column(query, PARTITION_COLUMN_ID)) is None:
            raise RuntimeError("partition column not found in query")

        ids_only = False
        if id_col is not None:
            query = query.with_only_columns(id_col, partition_col)
            ids_only = True

        current_partition: Optional[int] = None
        batch: list = []

        query_fields = [str(c.name) for c in query.selected_columns]
        id_column_idx = query_fields.index("sys__id")
        partition_column_idx = query_fields.index(PARTITION_COLUMN_ID)

        ordered_query = query.order_by(None).order_by(
            partition_col,
            *query._order_by_clauses,
        )

        with contextlib.closing(execute(ordered_query)) as rows:
            for row in rows:
                partition = row[partition_column_idx]
                if current_partition != partition:
                    current_partition = partition
                    if len(batch) > 0:
                        yield batch
                        batch = []
                batch.append(row[id_column_idx] if ids_only else row)

            if len(batch) > 0:
                yield batch
