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


lambda1 = lambda x: x * 2  # noqa: E731
lambda2 = lambda y: y + 1  # noqa: E731
lambda3 = lambda z: z - 1  # noqa: E731


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
            [
                func.row_number().over(
                    func.window(partition_by="file.name", order_by="file.name")
                )
            ],
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
    "func,expected_hash",
    [
        (double, "aba077bec793c25e277923cde6905636a80595d1cb9a92a2c53432fc620d2f44"),
        (
            double_arg_annot,
            "391b2bfe41cfb76a9bb7e72c5ab4333f89124cd256d87cee93378739d078400f",
        ),
        (
            double_arg_and_return_annot,
            "5f6c61c05d2c01a1b3745a69580cbf573ecdce2e09cce332cb83db0b270ff870",
        ),
    ],
)
def test_hash_named_functions(func, expected_hash):
    h = hash_callable(func)
    assert h == expected_hash


@pytest.mark.parametrize(
    "func",
    [
        lambda1,
        lambda2,
        lambda3,
    ],
)
def test_lambda_same_hash(func):
    h1 = hash_callable(func)
    h2 = hash_callable(func)
    assert h1 == h2  # same object produces same hash


def test_lambda_different_hashes():
    h1 = hash_callable(lambda1)
    h2 = hash_callable(lambda2)
    h3 = hash_callable(lambda3)

    # Ensure hashes are all different
    assert len({h1, h2, h3}) == 3
