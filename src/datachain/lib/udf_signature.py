import inspect
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Union, get_args, get_origin

from datachain.lib.data_model import DataType, DataTypeNames, is_chain_type
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.utils import AbstractUDF, DataChainParamsError


class UdfSignatureError(DataChainParamsError):
    def __init__(self, chain: str, msg):
        suffix = f"(dataset '{chain}')" if chain else ""
        super().__init__(f"processor signature error{suffix}: {msg}")


@dataclass
class UdfSignature:
    func: Callable
    params: Sequence[str]
    output_schema: SignalSchema

    DEFAULT_RETURN_TYPE = str

    @classmethod
    def parse(
        cls,
        chain: str,
        signal_map: dict[str, Callable],
        func: Optional[Callable] = None,
        params: Union[None, str, Sequence[str]] = None,
        output: Union[None, DataType, Sequence[str], dict[str, DataType]] = None,
        is_generator: bool = True,
    ) -> "UdfSignature":
        keys = ", ".join(signal_map.keys())
        if len(signal_map) > 1:
            raise UdfSignatureError(
                chain,
                f"multiple signals '{keys}' are not supported in processors."
                " Chain multiple processors instead.",
            )
        if len(signal_map) == 1:
            if func is not None:
                raise UdfSignatureError(
                    chain,
                    f"processor can't have signal '{keys}' with function '{func}'",
                )
            signal_name, udf_func = next(iter(signal_map.items()))
        else:
            if func is None:
                raise UdfSignatureError(chain, "user function is not defined")

            udf_func = func
            signal_name = None

        if not callable(udf_func):
            raise UdfSignatureError(chain, f"UDF '{udf_func}' is not callable")

        func_params_map_sign, func_outs_sign, is_iterator = (
            UdfSignature._func_signature(chain, udf_func)
        )
        if params:
            udf_params = [params] if isinstance(params, str) else params
        elif not func_params_map_sign:
            udf_params = []
        else:
            udf_params = list(func_params_map_sign.keys())
        if output:
            udf_output_map = UdfSignature._validate_output(
                chain, signal_name, func, func_outs_sign, output
            )
        else:
            if not func_outs_sign:
                raise UdfSignatureError(
                    chain,
                    f"outputs are not defined in function '{udf_func.__name__}'"
                    " hints or 'output'",
                )

            if not signal_name:
                raise UdfSignatureError(
                    chain,
                    "signal name is not specified."
                    " Define it as signal name 's1=func() or in 'output'",
                )

            if is_generator and not is_iterator:
                raise UdfSignatureError(
                    chain,
                    f"function '{func}' cannot be used in generator/aggregator"
                    " because it returns a type that is not Iterator/Generator."
                    f" Instead, it returns '{func_outs_sign}'",
                )

            if isinstance(func_outs_sign, tuple):
                udf_output_map = {
                    signal_name + f"_{num}": typ
                    for num, typ in enumerate(func_outs_sign)
                }
            else:
                udf_output_map = {signal_name: func_outs_sign[0]}

        return cls(
            func=udf_func,
            params=udf_params,
            output_schema=SignalSchema(udf_output_map),
        )

    @staticmethod
    def _validate_output(chain, signal_name, func, func_outs_sign, output):
        if isinstance(output, str):
            output = [output]
        if isinstance(output, Sequence):
            if len(func_outs_sign) != len(output):
                raise UdfSignatureError(
                    chain,
                    f"length of outputs names ({len(output)}) and function '{func}'"
                    f" return type length ({len(func_outs_sign)}) does not match",
                )

            udf_output_map = dict(zip(output, func_outs_sign))
        elif isinstance(output, dict):
            for key, value in output.items():
                if not isinstance(key, str):
                    raise UdfSignatureError(
                        chain,
                        f"output signal '{key}' has type '{type(key)}'"
                        " while 'str' is expected",
                    )
                if not is_chain_type(value):
                    raise UdfSignatureError(
                        chain,
                        f"output type '{value.__name__}' of signal '{key}' is not"
                        f" supported. Please use DataModel types: {DataTypeNames}",
                    )

            udf_output_map = output
        elif is_chain_type(output):
            udf_output_map = {signal_name: output}
        else:
            raise UdfSignatureError(
                chain,
                f"unknown output type: {output}. List of signals or dict of signals"
                " to function are expected.",
            )
        return udf_output_map

    def __eq__(self, other) -> bool:
        return (
            self.func == other.func
            and self.params == other.params
            and self.output_schema.values == other.output_schema.values
        )

    @staticmethod
    def _func_signature(
        chain: str, udf_func: Callable
    ) -> tuple[dict[str, type], Sequence[type], bool]:
        if isinstance(udf_func, AbstractUDF):
            func = udf_func.process  # type: ignore[unreachable]
        else:
            func = udf_func

        sign = inspect.signature(func)

        input_map = {prm.name: prm.annotation for prm in sign.parameters.values()}
        is_iterator = False

        anno = sign.return_annotation
        if anno == inspect.Signature.empty:
            output_types: list[type] = []
        else:
            orig = get_origin(anno)
            if inspect.isclass(orig) and issubclass(orig, Iterator):
                args = get_args(anno)
                if len(args) > 1 and not (
                    issubclass(orig, Generator) and len(args) == 3
                ):
                    raise UdfSignatureError(
                        chain,
                        f"function '{func}' should return iterator with a single"
                        f" value while '{args}' are specified",
                    )
                is_iterator = True
                anno = args[0]
                orig = get_origin(anno)

            if orig and orig is tuple:
                output_types = tuple(get_args(anno))  # type: ignore[assignment]
            else:
                output_types = [anno]

        if not output_types:
            output_types = [UdfSignature.DEFAULT_RETURN_TYPE]

        return input_map, output_types, is_iterator
