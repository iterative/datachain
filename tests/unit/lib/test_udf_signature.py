from collections.abc import Sequence
from typing import Callable, Optional, Union

import pytest

from datachain.lib.feature import Feature, FeatureType
from datachain.lib.file import File
from datachain.lib.udf import Mapper
from datachain.lib.udf_signature import UdfSignature, UdfSignatureError


def get_sign(
    func: Optional[Callable] = None,
    params: Union[None, str, Sequence[str]] = None,
    output: Union[None, FeatureType, Sequence[str], dict[str, FeatureType]] = None,
    **signal_map,
):
    return UdfSignature.parse("test", signal_map, func, params, output, False)


def func_str(p1) -> str:
    return "qwe"


def func_tuple(p1) -> tuple[Feature, str, int]:
    return File(name="n1"), "qwe", 33


def func_args(*args):
    return 12345


def test_basic():
    sign = get_sign(s1=func_str)

    assert sign.func == func_str
    assert sign.params == ["p1"]
    assert sign.output_schema.values == {"s1": str}


def test_basic_func():
    sign1 = get_sign(s1=func_str)
    sign2 = get_sign(func_str, output="s1")
    sign3 = get_sign(func_str, output=["s1"])
    sign4 = get_sign(s1=func_str, params="p1")
    sign5 = get_sign(s1=func_str, params=["p1"])

    assert sign1 == sign2
    assert sign1 == sign3
    assert sign1 == sign4
    assert sign1 == sign5


def test_signature_overwrite():
    sign = get_sign(s1=func_str, output={"my_sign": int}, params="some_prm")

    assert sign.func == func_str
    assert sign.params == ["some_prm"]
    assert sign.output_schema.values == {"my_sign": int}


def test_output_feature():
    sign = get_sign(s1=func_str, output={"my_sign": File})

    assert sign.output_schema.values == {"my_sign": File}


def test_output_as_value():
    sign = get_sign(s1=func_str, output="my_sign")

    assert sign.func == func_str
    assert sign.params == ["p1"]
    assert sign.output_schema.values == {"my_sign": str}


def test_output_as_list():
    sign = get_sign(s1=func_str, output=["my_sign"])

    assert sign.func == func_str
    assert sign.params == ["p1"]
    assert sign.output_schema.values == {"my_sign": str}


def test_multi_outputs_not_supported_yet():
    sign = get_sign(s1=func_tuple, output=["o1", "o2", "o3"])

    assert sign.output_schema.values == {"o1": Feature, "o2": str, "o3": int}


def test_multiple_signals_error():
    with pytest.raises(UdfSignatureError):
        get_sign(my_out=func_tuple, my_out2=func_str)

    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple, my_out=func_str)


def test_no_outputs():
    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple)

    with pytest.raises(UdfSignatureError):
        get_sign()


def test_tuple_output_number_mismatch():
    with pytest.raises(UdfSignatureError):
        get_sign(func_tuple, output=["a1", "a2", "a3", "a4", "a5"])


def test_no_params():
    with pytest.raises(UdfSignatureError):
        get_sign(lambda: 4)


def test_func_with_args():
    sign = get_sign(func_args, params=["prm1", "prm2"], output={"res": int})
    assert sign.params == ["prm1", "prm2"]


def test_output_type_error():
    with pytest.raises(UdfSignatureError):
        get_sign(func_str, output={"res": complex})

    with pytest.raises(UdfSignatureError):

        class TestCls:
            pass

        get_sign(func_str, output={"res": TestCls})


def test_feature_to_tuple_string_as_default_type():
    sign = get_sign(val1=lambda file: "asd")
    assert sign.output_schema.values == {"val1": str}


def test_callable_class():
    class MyTest:
        def __call__(self, file, p2) -> float:
            return 2.72

    sign = get_sign(s1=MyTest())
    assert sign.output_schema.values == {"s1": float}


def test_not_callable():
    class MyTest:
        def my_func(self, file, p2) -> float:
            return 2.72

    with pytest.raises(UdfSignatureError):
        get_sign(s1=MyTest())

    with pytest.raises(UdfSignatureError):
        get_sign(s1=123)


def test_udf_class():
    class MyTest(Mapper):
        def process(self, file, p2) -> int:
            return 42

    sign = get_sign(s1=MyTest())

    assert sign.output_schema.values == {"s1": int}
    assert sign.params == ["file", "p2"]


def test_udf_flatten_value():
    class MyTest(Mapper):
        def process(self, file, pp) -> int:
            return 42

    sign = get_sign(MyTest(), output={"res1": int})

    assert sign.output_schema.values == {"res1": int}


def test_udf_flatten_feature():
    class MyData(Feature):
        text: str
        count: int

    class MyTest(Mapper):
        def process(self, file, pp) -> MyData:
            return MyData(text="asdf", count=135)

    sign = get_sign(r1=MyTest())

    assert sign.output_schema.values == {"r1": MyData}
