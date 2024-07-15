import inspect
import sys
import traceback
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Callable, Optional

from fsspec.callbacks import DEFAULT_CALLBACK, Callback
from pydantic import BaseModel

from datachain.dataset import RowDict
from datachain.lib.data_model import FileBasic
from datachain.lib.feature import ModelUtil
from datachain.lib.feature_registry import Registry
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf_signature import UdfSignature
from datachain.lib.utils import AbstractUDF, DataChainError, DataChainParamsError
from datachain.query.batch import RowBatch
from datachain.query.schema import ColumnParameter
from datachain.query.udf import UDFBase as _UDFBase
from datachain.query.udf import UDFProperties, UDFResult

if TYPE_CHECKING:
    from typing_extensions import Self

    from datachain.catalog import Catalog
    from datachain.query.batch import BatchingResult


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
        udf_inputs: "Iterable[BatchingResult]",
        catalog: "Catalog",
        is_generator: bool,
        cache: bool,
        download_cb: Callback = DEFAULT_CALLBACK,
        processed_cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterator[Iterable["UDFResult"]]:
        self.inner._catalog = catalog
        if hasattr(self.inner, "setup") and callable(self.inner.setup):
            self.inner.setup()

        for batch in udf_inputs:
            n_rows = len(batch.rows) if isinstance(batch, RowBatch) else 1
            output = self.run_once(catalog, batch, is_generator, cache, cb=download_cb)
            processed_cb.relative_update(n_rows)
            yield output

        if hasattr(self.inner, "teardown") and callable(self.inner.teardown):
            self.inner.teardown()

    def run_once(
        self,
        catalog: "Catalog",
        arg: "BatchingResult",
        is_generator: bool = False,
        cache: bool = False,
        cb: Callback = DEFAULT_CALLBACK,
    ) -> Iterable[UDFResult]:
        if isinstance(arg, RowBatch):
            udf_inputs = [
                self.bind_parameters(catalog, row, cache=cache, cb=cb)
                for row in arg.rows
            ]
            udf_outputs = self.inner(udf_inputs, download_cb=cb)
            return self._process_results(arg.rows, udf_outputs, is_generator)
        if isinstance(arg, RowDict):
            udf_inputs = self.bind_parameters(catalog, arg, cache=cache, cb=cb)
            udf_outputs = self.inner(*udf_inputs, download_cb=cb)
            if not is_generator:
                # udf_outputs is generator already if is_generator=True
                udf_outputs = [udf_outputs]
            return self._process_results([arg], udf_outputs, is_generator)
        raise ValueError(f"Unexpected UDF argument: {arg}")


class UDFBase(AbstractUDF):
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

    def __call__(self, *rows, download_cb):
        if self.is_input_grouped:
            objs = self._parse_grouped_rows(rows[0], download_cb)
        else:
            objs = self._parse_rows(rows, download_cb)

        if not self.is_input_batched:
            objs = objs[0]

        result_objs = self.process_safe(objs)

        if not self.is_output_batched:
            result_objs = [result_objs]

        if len(self.output.values) > 1:
            res = []
            for tuple_ in result_objs:
                flat = []
                for obj in tuple_:
                    if isinstance(obj, BaseModel):
                        flat.extend(ModelUtil.flatten(obj))
                    else:
                        flat.append(obj)
                res.append(flat)
        else:
            # Generator expression is required, otherwise the value will be materialized
            res = (
                ModelUtil.flatten(obj)
                if isinstance(obj, BaseModel)
                else obj
                if isinstance(obj, tuple)
                else (obj,)
                for obj in result_objs
            )

        if not self.is_output_batched:
            res = list(res)
            assert len(res) == 1, (
                f"{self.name} returns {len(res)} " f"rows while it's not batched"
            )
            if isinstance(res[0], tuple):
                res = res[0]

        return res

    def _parse_rows(self, rows, download_cb):
        if not self.is_input_batched:
            rows = [rows]
        objs = []
        for row in rows:
            obj_row = self.params.row_to_objs(row)
            for obj in obj_row:
                if isinstance(obj, FileBasic):
                    obj._set_stream(
                        self._catalog, caching_enabled=True, download_cb=download_cb
                    )
            objs.append(obj_row)
        return objs

    def _parse_grouped_rows(self, group, download_cb):
        spec_map = {}
        output_map = {}
        for name, (anno, subtree) in self.params.tree.items():
            if inspect.isclass(anno) and issubclass(anno, BaseModel):
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

                if Registry.is_pydantic(cls):
                    obj = cls(**ModelUtil.unflatten_to_json(cls, slice))
                else:
                    obj = slice[0]

                if isinstance(obj, FileBasic):
                    obj._set_stream(self._catalog, download_cb=download_cb)
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
    pass


class BatchMapper(Mapper):
    is_input_batched = True
    is_output_batched = True


class Generator(UDFBase):
    is_output_batched = True


class Aggregator(UDFBase):
    is_input_batched = True
    is_output_batched = True
    is_input_grouped = True
