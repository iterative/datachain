import sys
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional

import attrs
from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from pydantic import BaseModel

from datachain.dataset import RowDict
from datachain.lib.convert.flatten import flatten
from datachain.lib.data_model import DataValue
from datachain.lib.file import File
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import AbstractUDF, DataChainError, DataChainParamsError
from datachain.query.batch import (
    Batch,
    BatchingStrategy,
    NoBatching,
    Partition,
    RowsOutputBatch,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from datachain.catalog import Catalog
    from datachain.lib.udf_signature import UdfSignature
    from datachain.query.batch import RowsOutput


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

    @property
    def signal_names(self) -> Iterable[str]:
        return self.output.keys()

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
        is_generator: bool,
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable[UDFResult]]:
        self.inner.catalog = catalog
        if hasattr(self.inner, "setup") and callable(self.inner.setup):
            self.inner.setup()

        for batch in udf_inputs:
            if isinstance(batch, RowsOutputBatch):
                n_rows = len(batch.rows)
                rows = [RowDict(zip(udf_fields, row)) for row in batch.rows]
                udf_inputs = [
                    self.bind_parameters(catalog, row, cache=cache, cb=download_cb)
                    for row in rows
                ]
                udf_outputs = self.inner.run_once(
                    udf_inputs, cache=cache, download_cb=download_cb
                )
                output = self._process_results(rows, udf_outputs, is_generator)
            else:
                n_rows = 1
                row = RowDict(zip(udf_fields, batch))
                udf_input = self.bind_parameters(
                    catalog, row, cache=cache, cb=download_cb
                )
                udf_outputs = self.inner.run_once(
                    udf_input, cache=cache, download_cb=download_cb
                )
                if not is_generator:
                    # udf_outputs is generator already if is_generator=True
                    udf_outputs = [udf_outputs]
                output = self._process_results([row], udf_outputs, is_generator)
            processed_cb.relative_update(n_rows)
            yield output

        if hasattr(self.inner, "teardown") and callable(self.inner.teardown):
            self.inner.teardown()

    def bind_parameters(self, catalog: "Catalog", row: "RowDict", **kwargs) -> list:
        assert self.inner.params
        return [row[p] for p in self.inner.params.to_udf_spec()]

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
        ]


class UDFBase(AbstractUDF):
    """Base class for stateful user-defined functions.

    Any class that inherits from it must have a `process()` method that takes input
    params from one or more rows in the chain and produces the expected output.

    Optionally, the class may include these methods:
    - `setup()` to run code on each  worker before `process()` is called.
    - `teardown()` to run code on each  worker after `process()` completes.

    Example:
        ```py
        from datachain import C, DataChain, Mapper
        import open_clip

        class ImageEncoder(Mapper):
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
            DataChain.from_storage(
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
    is_input_grouped = False
    catalog: "Optional[Catalog]"

    def __init__(self):
        self.params: Optional[SignalSchema] = None
        self.output = None
        self.catalog = None
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
        params: SignalSchema,
        func: Optional[Callable],
    ):
        self.params = params
        self.output = sign.output_schema
        self._func = func

    @classmethod
    def _create(
        cls,
        sign: "UdfSignature",
        params: SignalSchema,
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

    def to_udf_wrapper(self, batch: int = 1) -> UDFAdapter:
        return UDFAdapter(
            self,
            self.output.to_udf_spec(),
            batch,
        )

    def _pre_process(self, rows, cache, download_cb):
        return self._parse_rows([rows], cache, download_cb)[0]

    def run_once(self, rows, cache, download_cb):
        objs = self._pre_process(rows, cache, download_cb)
        result_objs = self.process_safe(objs)
        return self._post_process(result_objs, rows)

    def _post_process(self, result_objs, orig_rows):
        # Generator expression is required, otherwise the value will be materialized
        res = (self._flatten_row(row) for row in result_objs)

        if self.is_input_batched and not self.is_input_grouped:
            res = list(res)
            assert len(res) == len(
                orig_rows
            ), f"{self.name} returns {len(res)} rows while {len(orig_rows)} expected"

        return res

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

    def _parse_rows(
        self, rows, cache: bool, download_cb: Callback
    ) -> list[list[DataValue]]:
        assert self.params
        objs = []
        for row in rows:
            obj_row = self.params.row_to_objs(row)
            for obj in obj_row:
                if isinstance(obj, File):
                    assert self.catalog is not None
                    obj._set_stream(
                        self.catalog, caching_enabled=cache, download_cb=download_cb
                    )
            objs.append(obj_row)
        return objs

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


class Mapper(UDFBase):
    """Inherit from this class to pass to `DataChain.map()`."""

    def _post_process(self, result_objs, orig_rows):
        return self._flatten_row(result_objs)


class BatchMapper(UDFBase):
    """Inherit from this class to pass to `DataChain.batch_map()`."""

    is_input_batched = True
    is_output_batched = True

    def _pre_process(self, rows, cache, download_cb):
        return zip(*self._parse_rows(rows, cache, download_cb))


class Generator(UDFBase):
    """Inherit from this class to pass to `DataChain.gen()`."""

    is_output_batched = True


class Aggregator(UDFBase):
    """Inherit from this class to pass to `DataChain.agg()`."""

    is_input_batched = True
    is_output_batched = True
    is_input_grouped = True

    def _pre_process(self, rows, cache, download_cb):
        return zip(*self._parse_rows(rows, cache, download_cb))
