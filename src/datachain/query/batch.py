import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

from datachain.data_storage.schema import PARTITION_COLUMN_ID
from datachain.data_storage.warehouse import SELECT_BATCH_SIZE
from datachain.query.utils import get_query_column, get_query_id_column

if TYPE_CHECKING:
    from sqlalchemy import Select


@dataclass
class RowsOutputBatch:
    rows: Sequence[Sequence]


RowsOutput = Union[Sequence, RowsOutputBatch]


class BatchingStrategy(ABC):
    """BatchingStrategy provides means of batching UDF executions."""

    is_batching: bool

    @abstractmethod
    def __call__(
        self,
        execute: Callable,
        query: "Select",
        ids_only: bool = False,
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
        query: "Select",
        ids_only: bool = False,
    ) -> Generator[Sequence, None, None]:
        if ids_only:
            query = query.with_only_columns(get_query_id_column(query))
        return execute(query)


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
        query: "Select",
        ids_only: bool = False,
    ) -> Generator[RowsOutputBatch, None, None]:
        if ids_only:
            query = query.with_only_columns(get_query_id_column(query))

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

    is_batching = True

    def __call__(
        self,
        execute: Callable,
        query: "Select",
        ids_only: bool = False,
    ) -> Generator[RowsOutputBatch, None, None]:
        id_col = get_query_id_column(query)
        if (partition_col := get_query_column(query, PARTITION_COLUMN_ID)) is None:
            raise RuntimeError("partition column not found in query")

        if ids_only:
            query = query.with_only_columns(id_col, partition_col)

        current_partition: Optional[int] = None
        batch: list[Sequence] = []

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
                        yield RowsOutputBatch(batch)
                        batch = []
                batch.append([row[id_column_idx]] if ids_only else row)

            if len(batch) > 0:
                yield RowsOutputBatch(batch)
