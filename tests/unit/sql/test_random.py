from datachain.sql import select
from datachain.sql.functions import rand


def test_rand(warehouse):
    query = select(rand())
    result = tuple(warehouse.db.execute(query))
    assert isinstance(result[0][0], int)
