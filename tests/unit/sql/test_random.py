from datachain import func
from datachain.sql import select


def test_rand(warehouse):
    query = select(func.random.rand())
    result = tuple(warehouse.db.execute(query))
    assert isinstance(result[0][0], int)
