import sys
import traceback
from typing import TYPE_CHECKING, Callable, Optional

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from pydantic import BaseModel

from datachain.dataset import RowDict
from datachain.lib.convert.flatten import flatten
from datachain.lib.convert.unflatten import unflatten_to_json
from datachain.lib.file import File
from datachain.lib.model_store import ModelStore
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf_signature import UdfSignature
from datachain.lib.utils import AbstractUDF, DataChainError, DataChainParamsError
from datachain.query.batch import UDFInputBatch
from datachain.query.schema import ColumnParameter
from datachain.query.udf import UDFBase as _UDFBase
from datachain.query.udf import UDFProperties

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from typing_extensions import Self

    from datachain.catalog import Catalog
    from datachain.query.batch import RowsOutput, UDFInput
    from datachain.query.udf import UDFResult


class UdfError(DataChainParamsError):
    def __init__(self, msg):
        super().__init__(f"UDF error: {msg}")


class UDFAdapter(_UDFBase):
    def __init__(
        self,
        inner: "UDFBase",
        properties: UDFProperties,
    ):
        self.inner = inner
        super().__init__(properties)

    def run(
        self,
        udf_fields: "Sequence[str]",
        udf_inputs: "Iterable[RowsOutput]",
        catalog: "Catalog",
        is_generator: bool,
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> "Iterator[Iterable[UDFResult]]":
        self.inner._catalog = catalog
        if hasattr(self.inner, "setup") and callable(self.inner.setup):
            self.inner.setup()

        yield from super().run(
            udf_fields,
            udf_inputs,
            catalog,
            is_generator,
            cache,
            download_cb,
            processed_cb,
        )

        if hasattr(self.inner, "teardown") and callable(self.inner.teardown):
            self.inner.teardown()

    def run_once(
        self,
        catalog: "Catalog",
        arg: "UDFInput",
        is_generator: bool = False,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
    ) -> "Iterable[UDFResult]":
        if isinstance(arg, UDFInputBatch):
            udf_inputs = [
                self.bind_parameters(catalog, row, cache=cache, cb=cb)
                for row in arg.rows
            ]
            udf_outputs = self.inner(udf_inputs, cache=cache, download_cb=cb)
            return self._process_results(arg.rows, udf_outputs, is_generator)
        if isinstance(arg, RowDict):
            udf_inputs = self.bind_parameters(catalog, arg, cache=cache, cb=cb)
            udf_outputs = self.inner(*udf_inputs, cache=cache, download_cb=cb)
            if not is_generator:
                # udf_outputs is generator already if is_generator=True
                udf_outputs = [udf_outputs]
            return self._process_results([arg], udf_outputs, is_generator)
        raise ValueError(f"Unexpected UDF argument: {arg}")


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
    params_spec: Optional[list[str]]

    def __init__(self):
        self.params = None
        self.output = None
        self.params_spec = None
        self.output_spec = None
        self._contains_stream = None
        self._catalog = None
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
        sign: UdfSignature,
        params: SignalSchema,
        func: Callable,
    ):
        self.params = params
        self.output = sign.output_schema

        params_spec = self.params.to_udf_spec()
        self.params_spec = list(params_spec.keys())
        self.output_spec = self.output.to_udf_spec()

        self._func = func

    @classmethod
    def _create(
        cls,
        sign: UdfSignature,
        params: SignalSchema,
    ) -> "Self":
        if isinstance(sign.func, AbstractUDF):
            if not isinstance(sign.func, cls):  # type: ignore[unreachable]
                raise UdfError(
                    f"cannot create UDF: provided UDF '{sign.func.__name__}'"
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

    def set_catalog(self, catalog):
        self._catalog = catalog.copy(db=False)

    @property
    def catalog(self):
        return self._catalog

    def to_udf_wrapper(self, batch: int = 1) -> UDFAdapter:
        assert self.params_spec is not None
        properties = UDFProperties(
            [ColumnParameter(p) for p in self.params_spec], self.output_spec, batch
        )
        return UDFAdapter(self, properties)

    def validate_results(self, results, *args, **kwargs):
        return results

    def __call__(self, *rows, cache, download_cb):
        if self.is_input_grouped:
            objs = self._parse_grouped_rows(rows[0], cache, download_cb)
        elif self.is_input_batched:
            objs = zip(*self._parse_rows(rows[0], cache, download_cb))
        else:
            objs = self._parse_rows([rows], cache, download_cb)[0]

        result_objs = self.process_safe(objs)

        if not self.is_output_batched:
            result_objs = [result_objs]

        # Generator expression is required, otherwise the value will be materialized
        res = (self._flatten_row(row) for row in result_objs)

        if not self.is_output_batched:
            res = list(res)
            assert (
                len(res) == 1
            ), f"{self.name} returns {len(res)} rows while it's not batched"
            if isinstance(res[0], tuple):
                res = res[0]
        elif (
            self.is_input_batched
            and self.is_output_batched
            and not self.is_input_grouped
        ):
            res = list(res)
            assert len(res) == len(
                rows[0]
            ), f"{self.name} returns {len(res)} rows while len(rows[0]) expected"

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

    def _parse_rows(self, rows, cache, download_cb):
        objs = []
        for row in rows:
            obj_row = self.params.row_to_objs(row)
            for obj in obj_row:
                if isinstance(obj, File):
                    obj._set_stream(
                        self._catalog, caching_enabled=cache, download_cb=download_cb
                    )
            objs.append(obj_row)
        return objs

    def _parse_grouped_rows(self, group, cache, download_cb):
        spec_map = {}
        output_map = {}
        for name, (anno, subtree) in self.params.tree.items():
            if ModelStore.is_pydantic(anno):
                length = sum(1 for _ in self.params._get_flat_tree(subtree, [], 0))
            else:
                length = 1
            spec_map[name] = anno, length
            output_map[name] = []

        for flat_obj in group:
            position = 0
            for signal, (cls, length) in spec_map.items():
                slice = flat_obj[position : position + length]
                position += length

                if ModelStore.is_pydantic(cls):
                    obj = cls(**unflatten_to_json(cls, slice))
                else:
                    obj = slice[0]

                if isinstance(obj, File):
                    obj._set_stream(
                        self._catalog, caching_enabled=cache, download_cb=download_cb
                    )
                output_map[signal].append(obj)

        return list(output_map.values())

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


class BatchMapper(UDFBase):
    """Inherit from this class to pass to `DataChain.batch_map()`."""

    is_input_batched = True
    is_output_batched = True


class Generator(UDFBase):
    """Inherit from this class to pass to `DataChain.gen()`."""

    is_output_batched = True


class Aggregator(UDFBase):
    """Inherit from this class to pass to `DataChain.agg()`."""

    is_input_batched = True
    is_output_batched = True
    is_input_grouped = True
