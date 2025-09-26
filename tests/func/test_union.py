import datachain as dc


def test_union_persist_no_duplication_large_session(test_session, monkeypatch):
    # See https://github.com/iterative/datachain/issues/1356
    # Lower insert batch size to keep the test fast and still cross the boundary.
    monkeypatch.setattr(
        "datachain.data_storage.warehouse.INSERT_BATCH_SIZE", 20, raising=False
    )
    monkeypatch.setattr("datachain.query.dataset.INSERT_BATCH_SIZE", 20, raising=False)
    # Choose N large enough to cross the insert batch boundary
    n = 20 + 7

    x_ids = list(range(n))
    y_ids = list(range(n, 2 * n))

    x = dc.read_values(idx=x_ids, session=test_session)
    y = dc.read_values(idx=y_ids, session=test_session)

    # Ensure both branches have identical sys__id domains before union
    # This validates the original precondition that caused duplication on persist
    # when batching by sys__id and the ids overlapped across branches.
    # Access internal sys__id via the underlying DatasetQuery API
    x_sys_ids = {rec["sys__id"] for rec in x._query.to_db_records()}
    y_sys_ids = {rec["sys__id"] for rec in y._query.to_db_records()}
    # ClickHouse may generate sys__id independently per chain for read_values,
    # so the sets can differ there. Keep strict equality on other dialects and
    # enforce cardinality on ClickHouse.
    db_dialect = test_session.catalog.warehouse.db.dialect
    dialect_name = getattr(db_dialect, "name", None)
    if dialect_name == "clickhouse":
        assert len(x_sys_ids) == n
        assert len(y_sys_ids) == n
    else:
        assert x_sys_ids == y_sys_ids

    xy = x.union(y)
    assert xy.count() == 2 * n

    xy_p = xy.persist()
    assert xy_p.count() == 2 * n

    distinct_idx = {v for (v,) in xy_p.select("idx").results()}
    assert len(distinct_idx) == 2 * n

    j = xy_p.merge(x, on="idx", inner=True)
    assert j.count() == n


def test_union_parallel_udf_ids_only_no_dup(test_session_tmpfile, monkeypatch):
    # Validate that after union, running a parallel UDF that uses the ids-only
    # path does not duplicate rows due to sys__id collisions across branches.
    # This specifically exercises the parallel dispatch path where input rows
    # are split by sys__id and fetched per worker using IN (...) filters.
    # See https://github.com/iterative/datachain/issues/1356

    # Make worker/ids fetch batches small to exercise splitting on tiny inputs.
    monkeypatch.setattr("datachain.query.dispatch.DEFAULT_BATCH_SIZE", 5, raising=False)
    n = 30

    x_ids = list(range(n))
    y_ids = list(range(n, 2 * n))

    x = dc.read_values(idx=x_ids, session=test_session_tmpfile)
    y = dc.read_values(idx=y_ids, session=test_session_tmpfile)

    # Union and run a trivial UDF in parallel; parallel>1 forces the ids-only path
    # in the worker entrypoint, which would have amplified duplicates previously.
    xy = x.union(y)
    mapped = xy.settings(parallel=2).map(
        out=lambda idx: idx, output=int, params=("idx",)
    )

    # Count should match the number of input rows (2n) and equal distinct idx size.
    total = mapped.count()
    distinct_idx = {v for (v,) in mapped.select("idx").results()}

    assert total == 2 * n
    assert len(distinct_idx) == 2 * n
    assert total == len(distinct_idx)
