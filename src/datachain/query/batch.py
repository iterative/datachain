import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import sqlalchemy as sa

from datachain.data_storage.schema import PARTITION_COLUMN_ID
from datachain.data_storage.warehouse import SELECT_BATCH_SIZE

if TYPE_CHECKING:
    from datachain.dataset import RowDict


@dataclass
class RowBatch:
    rows: Sequence["RowDict"]


BatchingResult = Union["RowDict", RowBatch]


class BatchingStrategy(ABC):
    """BatchingStrategy provides means of batching UDF executions."""

    @abstractmethod
    def __call__(
        self,
        execute: Callable,
        query: sa.sql.selectable.Select,
    ) -> Generator[BatchingResult, None, None]:
        """Apply the provided parameters to the UDF."""


class NoBatching(BatchingStrategy):
    """
    NoBatching implements the default batching strategy, which is not to
    batch UDF calls.
    """

    def __call__(
        self,
        execute: Callable,
        query: sa.sql.selectable.Select,
    ) -> Generator["RowDict", None, None]:
        return execute(query, limit=query._limit, order_by=query._order_by_clauses)


class Batch(BatchingStrategy):
    """
    Batch implements UDF call batching, where each execution of a UDF
    is passed a sequence of multiple parameter sets.
    """

    def __init__(self, count: int):
        self.count = count

    def __call__(
        self,
        execute: Callable,
        query: sa.sql.selectable.Select,
    ) -> Generator[RowBatch, None, None]:
        # choose page size that is a multiple of the batch size
        page_size = math.ceil(SELECT_BATCH_SIZE / self.count) * self.count

        # select rows in batches
        results: list[RowDict] = []

        with contextlib.closing(
            execute(
                query,
                page_size=page_size,
                limit=query._limit,
                order_by=query._order_by_clauses,
            )
        ) as rows:
            for row in rows:
                results.append(row)
                if len(results) >= self.count:
                    batch, results = results[: self.count], results[self.count :]
                    yield RowBatch(batch)

            if len(results) > 0:
                yield RowBatch(results)


class Partition(BatchingStrategy):
    """
    Partition implements UDF call batching, where each execution of a UDF
    is run on a list of dataset rows grouped by the specified column.
    Dataset rows need to be sorted by the grouping column.
    """

    def __call__(
        self,
        execute: Callable,
        query: sa.sql.selectable.Select,
    ) -> Generator[RowBatch, None, None]:
        current_partition: Optional[int] = None
        batch: list[RowDict] = []

        with contextlib.closing(
            execute(
                query,
                order_by=(PARTITION_COLUMN_ID, "id", *query._order_by_clauses),
                limit=query._limit,
            )
        ) as rows:
            for row in rows:
                partition = row[PARTITION_COLUMN_ID]
                if current_partition != partition:
                    current_partition = partition
                    if len(batch) > 0:
                        yield RowBatch(batch)
                        batch = []
                batch.append(row)

            if len(batch) > 0:
                yield RowBatch(batch)
