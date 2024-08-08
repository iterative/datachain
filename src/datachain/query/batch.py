import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

from datachain.data_storage.schema import PARTITION_COLUMN_ID
from datachain.data_storage.warehouse import SELECT_BATCH_SIZE

if TYPE_CHECKING:
    from sqlalchemy import Select

    from datachain.dataset import RowDict


@dataclass
class RowsOutputBatch:
    rows: Sequence[Sequence]


RowsOutput = Union[Sequence, RowsOutputBatch]


@dataclass
class UDFInputBatch:
    rows: Sequence["RowDict"]


UDFInput = Union["RowDict", UDFInputBatch]


class BatchingStrategy(ABC):
    """BatchingStrategy provides means of batching UDF executions."""

    @abstractmethod
    def __call__(
        self,
        execute: Callable[..., Generator[Sequence, None, None]],
        query: "Select",
    ) -> Generator[RowsOutput, None, None]:
        """Apply the provided parameters to the UDF."""


class NoBatching(BatchingStrategy):
    """
    NoBatching implements the default batching strategy, which is not to
    batch UDF calls.
    """

    def __call__(
        self,
        execute: Callable[..., Generator[Sequence, None, None]],
        query: "Select",
    ) -> Generator[Sequence, None, None]:
        return execute(query)


class Batch(BatchingStrategy):
    """
    Batch implements UDF call batching, where each execution of a UDF
    is passed a sequence of multiple parameter sets.
    """

    def __init__(self, count: int):
        self.count = count

    def __call__(
        self,
        execute: Callable[..., Generator[Sequence, None, None]],
        query: "Select",
    ) -> Generator[RowsOutputBatch, None, None]:
        # choose page size that is a multiple of the batch size
        page_size = math.ceil(SELECT_BATCH_SIZE / self.count) * self.count

        # select rows in batches
        results: list[Sequence] = []

        with contextlib.closing(execute(query, page_size=page_size)) as rows:
            for row in rows:
                results.append(row)
                if len(results) >= self.count:
                    batch, results = results[: self.count], results[self.count :]
                    yield RowsOutputBatch(batch)

            if len(results) > 0:
                yield RowsOutputBatch(results)


class Partition(BatchingStrategy):
    """
    Partition implements UDF call batching, where each execution of a UDF
    is run on a list of dataset rows grouped by the specified column.
    Dataset rows need to be sorted by the grouping column.
    """

    def __call__(
        self,
        execute: Callable[..., Generator[Sequence, None, None]],
        query: "Select",
    ) -> Generator[RowsOutputBatch, None, None]:
        current_partition: Optional[int] = None
        batch: list[Sequence] = []

        query_fields = [str(c.name) for c in query.selected_columns]
        partition_column_idx = query_fields.index(PARTITION_COLUMN_ID)

        ordered_query = query.order_by(None).order_by(
            PARTITION_COLUMN_ID,
            "sys__id",
            *query._order_by_clauses,
        )

        with contextlib.closing(execute(ordered_query)) as rows:
            for row in rows:
                partition = row[partition_column_idx]
                if current_partition != partition:
                    current_partition = partition
                    if len(batch) > 0:
                        yield RowsOutputBatch(batch)
                        batch = []
                batch.append(row)

            if len(batch) > 0:
                yield RowsOutputBatch(batch)
