import pytest

from datachain.sql import select
from datachain.sql.selectable import values

DATA = [("a", 1.0, 100), ("b", 2.0, 200), ("c", 3.0, 300), ("d", 4.0, 400)]


@pytest.mark.parametrize(
    "query",
    [select(values(DATA)), select(values(DATA, ["letter", "float", "int"]))],
)
def test_select_values(warehouse, query):
    result = list(warehouse.db.execute(query))
    assert result == DATA


@pytest.mark.parametrize(
    "query",
    [
        select("c1", "c3").select_from(values(DATA)),
        select("letter", "int").select_from(values(DATA, ["letter", "float", "int"])),
    ],
)
def test_select_read_values(warehouse, query):
    result = list(warehouse.db.execute(query))
    assert result == [("a", 100), ("b", 200), ("c", 300), ("d", 400)]
