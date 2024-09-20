import typing
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

from fsspec.callbacks import DEFAULT_CALLBACK, Callback

from datachain.dataset import RowDict

from .batch import (
    Batch,
    BatchingStrategy,
    NoBatching,
    Partition,
    RowsOutputBatch,
    UDFInputBatch,
)
from .schema import UDFParameter

if TYPE_CHECKING:
    from datachain.catalog import Catalog

    from .batch import RowsOutput, UDFInput

ColumnType = Any


# Specification for the output of a UDF
UDFOutputSpec = typing.Mapping[str, ColumnType]

# Result type when calling the UDF wrapper around the actual
# Python function / class implementing it.
UDFResult = dict[str, Any]


@dataclass
class UDFProperties:
    """Container for basic UDF properties."""

    params: list[UDFParameter]
    output: UDFOutputSpec
    batch: int = 1

    def get_batching(self, use_partitioning: bool = False) -> BatchingStrategy:
        if use_partitioning:
            return Partition()
        if self.batch == 1:
            return NoBatching()
        if self.batch > 1:
            return Batch(self.batch)
        raise ValueError(f"invalid batch size {self.batch}")

    def signal_names(self) -> Iterable[str]:
        return self.output.keys()


class UDFBase:
    """A base class for implementing stateful UDFs."""

    def __init__(
        self,
        properties: UDFProperties,
    ):
        self.properties = properties
        self.signal_names = properties.signal_names()
        self.output = properties.output

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[RowsOutput]",
        catalog: "Catalog",
        is_generator: bool,
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable["UDFResult"]]:
        for batch in udf_inputs:
            if isinstance(batch, RowsOutputBatch):
                n_rows = len(batch.rows)
                inputs: UDFInput = UDFInputBatch(
                    [RowDict(zip(udf_fields, row)) for row in batch.rows]
                )
            else:
                n_rows = 1
                inputs = RowDict(zip(udf_fields, batch))
            output = self.run_once(catalog, inputs, is_generator, cache, cb=download_cb)
            processed_cb.relative_update(n_rows)
            yield output

    def run_once(
        self,
        catalog: "Catalog",
        arg: "UDFInput",
        is_generator: bool = False,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterable[UDFResult]:
        raise NotImplementedError

    def bind_parameters(self, catalog: "Catalog", row: "RowDict", **kwargs) -> list:
        return [p.get_value(catalog, row, **kwargs) for p in self.properties.params]

    def _process_results(
        self,
        rows: Sequence["RowDict"],
        results: Sequence[Sequence[Any]],
        is_generator=False,
    ) -> Iterable[UDFResult]:
        """Create a list of dictionaries representing UDF results."""

        # outputting rows
        if is_generator:
            # each row in results is a tuple of column values
            return (dict(zip(self.signal_names, row)) for row in results)

        # outputting signals
        row_ids = [row["sys__id"] for row in rows]
        return [
            {"sys__id": row_id} | dict(zip(self.signal_names, signals))
            for row_id, signals in zip(row_ids, results)
            if signals is not None  # skip rows with no output
        ]
