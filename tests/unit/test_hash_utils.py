import pytest
import sqlalchemy as sa

from datachain import C, func
from datachain.hash_utils import hash_column_elements


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
