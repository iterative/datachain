import pytest
from sqlalchemy import Integer

from datachain.dataset import RowDict
from datachain.query import udf
from datachain.query.batch import RowBatch
from datachain.query.schema import ColumnParameter


def test_udf_single_signal():
    @udf(("id", "size"), {"mul": Integer})
    def t(a, b):
        return (a * b,)

    row = RowDict(
        id=6,
        vtype="",
        dir_type=1,
        parent="",
        name="obj",
        last_modified=None,
        etag="",
        version="",
        is_latest=True,
        size=7,
        owner_name="",
        owner_id="",
        source="",
        random=1234,
        location=None,
    )
    result = t(None, row)
    assert result[0]["mul"] == (42)


def test_udf_multiple_signals():
    @udf(("id", "size"), {"mul": Integer, "sum": Integer})
    def t(a, b):
        return (a * b, a + b)

    row = RowDict(
        id=6,
        vtype="",
        dir_type=1,
        parent="",
        name="obj",
        last_modified=None,
        etag="",
        version="",
        is_latest=True,
        size=7,
        owner_name="",
        owner_id="",
        source="",
        random=1234,
        location=None,
    )
    result = t(None, row)
    assert result[0] == {"id": 6, "mul": 42, "sum": 13}


def test_udf_batching():
    @udf(("id", "size"), {"mul": Integer}, batch=4)
    def t(vals):
        return [(a * b,) for (a, b) in vals]

    inputs = list(zip(range(1, 11), range(21, 31)))
    results = []
    for size, row_id in inputs:
        row = RowDict(
            id=row_id,
            vtype="",
            dir_type=1,
            parent="",
            name="obj",
            last_modified=None,
            etag="",
            version="",
            is_latest=True,
            size=size,
            owner_name="",
            owner_id="",
            source="",
            random=1234,
            location=None,
        )
        batch = RowBatch([row])
        result = t(None, batch)
        if result:
            assert len(result) == 1  # Matches batch size.
            results.extend(result)

    assert len(results) == len(inputs)
    assert results == [{"id": b, "mul": a * b} for (a, b) in inputs]


def test_stateful_udf():
    @udf(("size",), {"sum": Integer}, method="sum")
    class MyUDF:
        def __init__(self, constant):
            self.constant = constant

        def sum(self, size):
            return (self.constant + size,)

    udf_inst = MyUDF(5)()
    inputs = range(1, 11)
    results = []
    for size in inputs:
        row = RowDict(
            id=5,
            vtype="",
            dir_type=1,
            parent="",
            name="obj",
            last_modified=None,
            etag="",
            version="",
            is_latest=True,
            size=size,
            owner_name="",
            owner_id="",
            source="",
            random=1234,
            location=None,
        )
        results.extend(udf_inst(None, row))

    assert len(results) == len(inputs)
    assert results == [{"id": 5, "sum": 5 + size} for size in inputs]


@pytest.mark.parametrize("param", ["foo", ("foo",)])
def test_udf_api(param):
    func = lambda x: x  # noqa: E731
    result = udf(param, {"bar": Integer}, batch=42)(func)
    assert result.func is func
    assert result.properties.params == [ColumnParameter("foo")]
    assert result.properties.output == {"bar": Integer}
    assert result.properties.batch == 42


def test_udf_error():
    with pytest.raises(TypeError):

        @udf(params=("name",), output=("name_len",))
        def name_len(name):
            return len(name)
