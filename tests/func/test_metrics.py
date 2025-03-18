from datachain.query import metrics


def test_metrics(test_session, catalog, mocker, monkeypatch):
    mocker.patch("datachain.query.session.Session.get", return_value=test_session)
    mocker.patch.dict("datachain.query.metrics.metrics", {}, clear=True)
    job_id = catalog.metastore.create_job("test_metrics", "")
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)

    metrics.set("foo", 42)

    assert metrics.get("foo") == 42
    (job,) = catalog.metastore.list_jobs_by_ids([job_id])
    assert job.metrics == metrics.metrics == {"foo": 42}
