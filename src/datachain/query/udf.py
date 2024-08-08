import typing
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import WRAPPER_ASSIGNMENTS
from inspect import isclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
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
from .schema import (
    UDFParameter,
    UDFParamSpec,
    normalize_param,
)

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


def udf(
    params: Sequence[UDFParamSpec],
    output: UDFOutputSpec,
    *,
    method: Optional[str] = None,  # only used for class-based UDFs
    batch: int = 1,
):
    """
    Decorate a function or a class to be used as a UDF.

    The decorator expects both the outputs and inputs of the UDF to be specified.
    The outputs are defined as a collection of tuples containing the signal name
    and type.
    Parameters are defined as a list of column objects (e.g. C.name).
    Optionally, UDFs can be run on batches of rows to improve performance, this
    is determined by the 'batch' parameter. When operating on batches of inputs,
    the UDF function will be called with a single argument - a list
    of tuples containing inputs (e.g. ((input1_a, input1_b), (input2_a, input2b))).
    """
    if isinstance(params, str):
        params = (params,)
    if not isinstance(output, Mapping):
        raise TypeError(f"'output' must be a mapping, got {type(output).__name__}")

    properties = UDFProperties([normalize_param(p) for p in params], output, batch)

    def decorator(udf_base: Union[Callable, type]):
        if isclass(udf_base):
            return UDFClassWrapper(udf_base, properties, method=method)
        if callable(udf_base):
            return UDFWrapper(udf_base, properties)

    return decorator


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


class UDFClassWrapper:
    """
    A wrapper for class-based (stateful) UDFs.
    """

    def __init__(
        self,
        udf_class: type,
        properties: UDFProperties,
        method: Optional[str] = None,
    ):
        self.udf_class = udf_class
        self.udf_method = method
        self.properties = properties
        self.output = properties.output

    def __call__(self, *args, **kwargs) -> "UDFFactory":
        return UDFFactory(
            self.udf_class,
            args,
            kwargs,
            self.properties,
            self.udf_method,
        )


class UDFWrapper(UDFBase):
    """A wrapper class for function UDFs to be used in custom signal generation."""

    def __init__(
        self,
        func: Callable,
        properties: UDFProperties,
    ):
        self.func = func
        super().__init__(properties)
        # This emulates the behavior of functools.wraps for a class decorator
        for attr in WRAPPER_ASSIGNMENTS:
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def run_once(
        self,
        catalog: "Catalog",
        arg: "UDFInput",
        is_generator: bool = False,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterable[UDFResult]:
        if isinstance(arg, UDFInputBatch):
            udf_inputs = [
                self.bind_parameters(catalog, row, cache=cache, cb=cb)
                for row in arg.rows
            ]
            udf_outputs = self.func(udf_inputs)
            return self._process_results(arg.rows, udf_outputs, is_generator)
        if isinstance(arg, RowDict):
            udf_inputs = self.bind_parameters(catalog, arg, cache=cache, cb=cb)
            udf_outputs = self.func(*udf_inputs)
            if not is_generator:
                # udf_outputs is generator already if is_generator=True
                udf_outputs = [udf_outputs]
            return self._process_results([arg], udf_outputs, is_generator)
        raise ValueError(f"Unexpected UDF argument: {arg}")

    # This emulates the behavior of functools.wraps for a class decorator
    def __repr__(self):
        return repr(self.func)


class UDFFactory:
    """
    A wrapper for late instantiation of UDF classes, primarily for use in parallelized
    execution.
    """

    def __init__(
        self,
        udf_class: type,
        args,
        kwargs,
        properties: UDFProperties,
        method: Optional[str] = None,
    ):
        self.udf_class = udf_class
        self.udf_method = method
        self.args = args
        self.kwargs = kwargs
        self.properties = properties
        self.output = properties.output

    def __call__(self) -> UDFWrapper:
        udf_func = self.udf_class(*self.args, **self.kwargs)
        if self.udf_method:
            udf_func = getattr(udf_func, self.udf_method)

        return UDFWrapper(udf_func, self.properties)


UDFType = Union[UDFBase, UDFFactory]
