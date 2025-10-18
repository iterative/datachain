import datachain as dc
from datachain.query.dataset import DatasetQuery


def _capture_temp_tables(mocker):
    captured: list[list[str]] = []
    original_cleanup = DatasetQuery.cleanup

    def capture(self):
        captured.append(list(self.temp_table_names))
        return original_cleanup(self)

    mocker.patch("datachain.query.dataset.DatasetQuery.cleanup", capture)
    return captured


def _assert_no_duplicate_temp_tables(captured: list[list[str]]):
    for tables in captured:
        assert len(tables) == len(set(tables))


def test_nested_merge_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    base = dc.read_values(num=[1, 2], session=test_session)
    generated = base.map(num_plus=lambda num: str(num + 10))
    inner = generated.merge(base, on="num", inner=True)
    chain = base.merge(inner, on="num", inner=True)

    expected = chain.select("num").to_pandas()["num"].tolist()
    assert expected == [1, 2]

    rerun = chain.select("num").to_pandas()["num"].tolist()
    assert rerun == expected

    _assert_no_duplicate_temp_tables(captured)


def test_union_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    left = dc.read_values(num=[1, 2], session=test_session)
    right = dc.read_values(num=[3], session=test_session)
    union_chain = left.union(right)

    expected = sorted(union_chain.select("num").to_pandas()["num"].tolist())
    assert expected == [1, 2, 3]

    rerun = sorted(union_chain.select("num").to_pandas()["num"].tolist())
    assert rerun == expected

    _assert_no_duplicate_temp_tables(captured)


def test_subtract_has_no_duplicate_temp_tables(test_session, mocker):
    captured = _capture_temp_tables(mocker)

    source = dc.read_values(num=[1, 2], session=test_session)
    target = dc.read_values(num=[2], session=test_session)
    subtract_chain = source.subtract(target, on="num")

    expected = subtract_chain.select("num").to_pandas()["num"].tolist()
    assert expected == [1]

    rerun = subtract_chain.select("num").to_pandas()["num"].tolist()
    assert rerun == expected

    _assert_no_duplicate_temp_tables(captured)
