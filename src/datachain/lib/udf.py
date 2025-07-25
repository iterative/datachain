import sys
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing, nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import attrs
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from pydantic import BaseModel

from datachain.asyn import AsyncMapper
from datachain.cache import temporary_cache
from datachain.dataset import RowDict
from datachain.lib.convert.flatten import flatten
from datachain.lib.file import DataModel, File
from datachain.lib.utils import AbstractUDF, DataChainError, DataChainParamsError
from datachain.query.batch import (
    Batch,
    BatchingStrategy,
    NoBatching,
    Partition,
    RowsOutputBatch,
)
from datachain.utils import safe_closing

if TYPE_CHECKING:
    from collections import abc
    from contextlib import AbstractContextManager

    from typing_extensions import Self

    from datachain.cache import Cache
    from datachain.catalog import Catalog
    from datachain.lib.signal_schema import SignalSchema
    from datachain.lib.udf_signature import UdfSignature
    from datachain.query.batch import RowsOutput

T = TypeVar("T", bound=Sequence[Any])


class UdfError(DataChainParamsError):
    def __init__(self, msg):
        super().__init__(f"UDF error: {msg}")


ColumnType = Any

# Specification for the output of a UDF
UDFOutputSpec = Mapping[str, ColumnType]

# Result type when calling the UDF wrapper around the actual
# Python function / class implementing it.
UDFResult = dict[str, Any]


@attrs.define
class UDFProperties:
    udf: "UDFAdapter"

    def get_batching(self, use_partitioning: bool = False) -> BatchingStrategy:
        return self.udf.get_batching(use_partitioning)

    @property
    def batch(self):
        return self.udf.batch


@attrs.define(slots=False)
class UDFAdapter:
    inner: "UDFBase"
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

    @property
    def properties(self):
        # For backwards compatibility.
        return UDFProperties(self)

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[RowsOutput]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        yield from self.inner.run(
            udf_fields,
            udf_inputs,
            catalog,
            cache,
            download_cb,
            processed_cb,
        )

    @property
    def prefetch(self) -> int:
        return self.inner.prefetch


class UDFBase(AbstractUDF):
    """Base class for stateful user-defined functions.

    Any class that inherits from it must have a `process()` method that takes input
    params from one or more rows in the chain and produces the expected output.

    Optionally, the class may include these methods:
    - `setup()` to run code on each  worker before `process()` is called.
    - `teardown()` to run code on each  worker after `process()` completes.

    Example:
        ```py
        import datachain as dc
        import open_clip

        class ImageEncoder(dc.Mapper):
            def __init__(self, model_name: str, pretrained: str):
                self.model_name = model_name
                self.pretrained = pretrained

            def setup(self):
                self.model, _, self.preprocess = (
                    open_clip.create_model_and_transforms(
                        self.model_name, self.pretrained
                    )
                )

            def process(self, file) -> list[float]:
                img = file.get_value()
                img = self.preprocess(img).unsqueeze(0)
                emb = self.model.encode_image(img)
                return emb[0].tolist()

        (
            dc.read_storage(
                "gs://datachain-demo/fashion-product-images/images", type="image"
            )
            .limit(5)
            .map(
                ImageEncoder("ViT-B-32", "laion2b_s34b_b79k"),
                params=["file"],
                output={"emb": list[float]},
            )
            .show()
        )
        ```
    """

    is_input_batched = False
    is_output_batched = False
    prefetch: int = 0

    def __init__(self):
        self.params: Optional[SignalSchema] = None
        self.output = None
        self._func = None

    def process(self, *args, **kwargs):
        """Processing function that needs to be defined by user"""
        if not self._func:
            raise NotImplementedError("UDF processing is not implemented")
        return self._func(*args, **kwargs)

    def setup(self):
        """Initialization process executed on each worker before processing begins.
        This is needed for tasks like pre-loading ML models prior to scoring.
        """

    def teardown(self):
        """Teardown process executed on each process/worker after processing ends.
        This is needed for tasks like closing connections to end-points.
        """

    def _init(
        self,
        sign: "UdfSignature",
        params: "SignalSchema",
        func: Optional[Callable],
    ):
        self.params = params
        self.output = sign.output_schema
        self._func = func

    @classmethod
    def _create(
        cls,
        sign: "UdfSignature",
        params: "SignalSchema",
    ) -> "Self":
        if isinstance(sign.func, AbstractUDF):
            if not isinstance(sign.func, cls):  # type: ignore[unreachable]
                raise UdfError(
                    f"cannot create UDF: provided UDF '{type(sign.func).__name__}'"
                    f" must be a child of target class '{cls.__name__}'",
                )
            result = sign.func
            func = None
        else:
            result = cls()
            func = sign.func

        result._init(sign, params, func)
        return result

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def verbose_name(self):
        """Returns the name of the function or class that implements the UDF."""
        if self._func and callable(self._func):
            if hasattr(self._func, "__name__"):
                return self._func.__name__
            if hasattr(self._func, "__class__") and hasattr(
                self._func.__class__, "__name__"
            ):
                return self._func.__class__.__name__
        return "<unknown>"

    @property
    def signal_names(self) -> Iterable[str]:
        return self.output.to_udf_spec().keys()

    def to_udf_wrapper(self, batch: int = 1) -> UDFAdapter:
        return UDFAdapter(
            self,
            self.output.to_udf_spec(),
            batch,
        )

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Any]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        raise NotImplementedError

    def _flatten_row(self, row):
        if len(self.output.values) > 1 and not isinstance(row, BaseModel):
            flat = []
            for obj in row:
                flat.extend(self._obj_to_list(obj))
            return tuple(flat)
        return row if isinstance(row, tuple) else tuple(self._obj_to_list(row))

    @staticmethod
    def _obj_to_list(obj):
        return flatten(obj) if isinstance(obj, BaseModel) else [obj]

    def _parse_row(
        self, row_dict: RowDict, catalog: "Catalog", cache: bool, download_cb: Callback
    ) -> list[Any]:
        assert self.params
        row = [row_dict[p] for p in self.params.to_udf_spec()]
        obj_row = self.params.row_to_objs(row)
        for obj in obj_row:
            self._set_stream_recursive(obj, catalog, cache, download_cb)
        return obj_row

    def _set_stream_recursive(
        self, obj: Any, catalog: "Catalog", cache: bool, download_cb: Callback
    ) -> None:
        """Recursively set the catalog stream on all File objects within an object."""
        if isinstance(obj, File):
            obj._set_stream(catalog, caching_enabled=cache, download_cb=download_cb)

        # Check all fields for nested File objects, but only for DataModel objects
        if isinstance(obj, DataModel):
            for field_name in type(obj).model_fields:
                field_value = getattr(obj, field_name, None)
                if isinstance(field_value, DataModel):
                    self._set_stream_recursive(field_value, catalog, cache, download_cb)

    def _prepare_row(self, row, udf_fields, catalog, cache, download_cb):
        row_dict = RowDict(zip(udf_fields, row))
        return self._parse_row(row_dict, catalog, cache, download_cb)

    def _prepare_row_and_id(self, row, udf_fields, catalog, cache, download_cb):
        row_dict = RowDict(zip(udf_fields, row))
        udf_input = self._parse_row(row_dict, catalog, cache, download_cb)
        return row_dict["sys__id"], *udf_input

    def process_safe(self, obj_rows):
        try:
            result_objs = self.process(*obj_rows)
        except Exception as e:  # noqa: BLE001
            msg = f"============== Error in user code: '{self.name}' =============="
            print(msg)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback.tb_next)
            print("=" * len(msg))
            raise DataChainError(
                f"Error in user code in class '{self.name}': {e!s}"
            ) from None
        return result_objs


def noop(*args, **kwargs):
    pass


async def _prefetch_input(
    row: T,
    download_cb: Optional["Callback"] = None,
    after_prefetch: "Callable[[], None]" = noop,
) -> T:
    for obj in row:
        if isinstance(obj, File) and obj.path and await obj._prefetch(download_cb):
            after_prefetch()
    return row


def _remove_prefetched(row: T) -> None:
    for obj in row:
        if isinstance(obj, File):
            catalog = obj._catalog
            assert catalog is not None
            try:
                catalog.cache.remove(obj)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to remove prefetched item {obj.name!r}: {e!s}")


def _prefetch_inputs(
    prepared_inputs: "Iterable[T]",
    prefetch: int = 0,
    download_cb: Optional["Callback"] = None,
    after_prefetch: Optional[Callable[[], None]] = None,
    remove_prefetched: bool = False,
) -> "abc.Generator[T, None, None]":
    if not prefetch:
        yield from prepared_inputs
        return

    if after_prefetch is None:
        after_prefetch = noop
        if download_cb and hasattr(download_cb, "increment_file_count"):
            increment_file_count: Callable[[], None] = download_cb.increment_file_count
            after_prefetch = increment_file_count

    f = partial(_prefetch_input, download_cb=download_cb, after_prefetch=after_prefetch)
    mapper = AsyncMapper(f, prepared_inputs, workers=prefetch)
    with closing(mapper.iterate()) as row_iter:
        for row in row_iter:
            try:
                yield row  # type: ignore[misc]
            finally:
                if remove_prefetched:
                    _remove_prefetched(row)


def _get_cache(
    cache: "Cache", prefetch: int = 0, use_cache: bool = False
) -> "AbstractContextManager[Cache]":
    tmp_dir = cache.tmp_dir
    assert tmp_dir
    if prefetch and not use_cache:
        return temporary_cache(tmp_dir, prefix="prefetch-")
    return nullcontext(cache)


class Mapper(UDFBase):
    """Inherit from this class to pass to `DataChain.map()`."""

    prefetch: int = 2

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Sequence[Any]]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.setup()

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    yield self._prepare_row_and_id(
                        row, udf_fields, catalog, cache, download_cb
                    )

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            self.prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(self.prefetch) and not cache,
        )

        with closing(prepared_inputs):
            for id_, *udf_args in prepared_inputs:
                result_objs = self.process_safe(udf_args)
                udf_output = self._flatten_row(result_objs)
                output = [{"sys__id": id_} | dict(zip(self.signal_names, udf_output))]
                processed_cb.relative_update(1)
                yield output

        self.teardown()


class BatchMapper(UDFBase):
    """Inherit from this class to pass to `DataChain.batch_map()`."""

    is_input_batched = True
    is_output_batched = True

    def run(
        self,
        udf_fields: Sequence[str],
        udf_inputs: Iterable[RowsOutputBatch],
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.setup()

        for batch in udf_inputs:
            n_rows = len(batch)
            row_ids, *udf_args = zip(
                *[
                    self._prepare_row_and_id(
                        row, udf_fields, catalog, cache, download_cb
                    )
                    for row in batch
                ]
            )
            result_objs = list(self.process_safe(udf_args))
            n_objs = len(result_objs)
            assert n_objs == n_rows, (
                f"{self.name} returns {n_objs} rows, but {n_rows} were expected"
            )
            udf_outputs = (self._flatten_row(row) for row in result_objs)
            output = [
                {"sys__id": row_id} | dict(zip(self.signal_names, signals))
                for row_id, signals in zip(row_ids, udf_outputs)
            ]
            processed_cb.relative_update(n_rows)
            yield output

        self.teardown()


class Generator(UDFBase):
    """Inherit from this class to pass to `DataChain.gen()`."""

    is_output_batched = True
    prefetch: int = 2

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[Sequence[Any]]",
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.setup()

        def _prepare_rows(udf_inputs) -> "abc.Generator[Sequence[Any], None, None]":
            with safe_closing(udf_inputs):
                for row in udf_inputs:
                    yield self._prepare_row(
                        row, udf_fields, catalog, cache, download_cb
                    )

        def _process_row(row):
            with safe_closing(self.process_safe(row)) as result_objs:
                for result_obj in result_objs:
                    udf_output = self._flatten_row(result_obj)
                    yield dict(zip(self.signal_names, udf_output))

        prepared_inputs = _prepare_rows(udf_inputs)
        prepared_inputs = _prefetch_inputs(
            prepared_inputs,
            self.prefetch,
            download_cb=download_cb,
            remove_prefetched=bool(self.prefetch) and not cache,
        )
        with closing(prepared_inputs):
            for row in prepared_inputs:
                yield _process_row(row)
                processed_cb.relative_update(1)

        self.teardown()


class Aggregator(UDFBase):
    """Inherit from this class to pass to `DataChain.agg()`."""

    is_input_batched = True
    is_output_batched = True

    def run(
        self,
        udf_fields: Sequence[str],
        udf_inputs: Iterable[RowsOutputBatch],
        catalog: "Catalog",
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.setup()

        for batch in udf_inputs:
            udf_args = zip(
                *[
                    self._prepare_row(row, udf_fields, catalog, cache, download_cb)
                    for row in batch
                ]
            )
            result_objs = self.process_safe(udf_args)
            udf_outputs = (self._flatten_row(row) for row in result_objs)
            output = (dict(zip(self.signal_names, row)) for row in udf_outputs)
            processed_cb.relative_update(len(batch))
            yield output

        self.teardown()
