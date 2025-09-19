import pytest
import sqlalchemy as sa

from datachain import C, func
from datachain.hash_utils import hash_callable, hash_column_elements


def double(x):
    return x * 2


def double_arg_annot(x: int):
    return x * 2


def double_arg_and_return_annot(x: int) -> int:
    return x * 2


@pytest.mark.parametrize(
    "expr,result",
    [
        (
            [C("name")],
            "8e95b415d0950727bb698f1a9fcaf28a4e088afe19d8256ecaa581022cde9365",
        ),
        (
            [C("name"), C("age")],
            "c4f98a6350d621d16255490fe0d522b61749720a27f6a42b262090bad4100092",
        ),
        (
            [func.avg("age")],
            "ddc23abe88c722954e568f7db548ddcbd060eed1a1a815bfcaabd1dce8add3aa",
        ),
        (
            [func.row_number().over(C("age"))],
            "9da0e1581399e92f628c00879422835fc05ada2584e9962c0edb20f87637e8bf",
        ),
        (
            [C("age").label("user_age")],
            "8a0a3d4e99972dc5fdc462b9981b309bbc6b0cc86d73880d56108ef0553bd426",
        ),
        (
            [C("age") > 20],
            "6ba1c4384c710fe439e84749d7d08d675cb03d4c3683eb55bce11efd42372b67",
        ),
        (
            [sa.and_(C("age") > 20, C("name") != "")],
            "a27c392ad1c294783ab70175478bf7cf2110fe559bf68504026f773e5aa361ab",
        ),
        (
            [],
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        ),
    ],
)
def test_hash_column_elements(expr, result):
    assert hash_column_elements(expr) == result


@pytest.mark.parametrize(
    "func,result",
    [
        (
            lambda x: x * 2,
            "da39989827ad84f945c7352676ff0b9ad388888ab5d1aa7d0b7dcbc3def83f11",
        ),
        (
            double,
            "85e734c651a38c659c5cd1ff21df3fa015427613354a2a5c24b15781227b30ad",
        ),
        (
            double_arg_annot,
            "bcd3314b4b2c08a171f1ee9c8d7c6434c052f46ab98136ca8262fc9cea6fb29a",
        ),
        (
            double_arg_and_return_annot,
            "77b4f4159e8f83586dbd09e0665c36155030696bbd0053539f7cab6bf7a32e05",
        ),
    ],
)
def test_hash_callable(func, result):
    assert hash_callable(func) == result
