import pytest

import datachain as dc
from datachain.error import DatasetNotFoundError
from datachain.lib.utils import DataChainError


def mapper_fail(num) -> int:
    raise Exception("Error")


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints(test_session, monkeypatch, nums_dataset, reset_checkpoints):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    # -------------- FIRST RUN -------------------
    first_job_id = catalog.metastore.create_job("my-job", "echo 1;")
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).save("nums2")
    with pytest.raises(DataChainError):
        (
            dc.read_dataset("nums", session=test_session)
            .map(new=mapper_fail)
            .save("nums3")
        )
    # check we created first 2 datasets but third one failed
    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    second_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=first_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).save("nums2")
    dc.read_dataset("nums", session=test_session).save("nums3")

    # check that we skipped creation of first 2 datasets (they have only 1 version)
    assert (
        len(catalog.get_dataset("nums1").versions) == 1 if not reset_checkpoints else 2
    )
    assert (
        len(catalog.get_dataset("nums2").versions) == 1 if not reset_checkpoints else 2
    )
    assert len(catalog.get_dataset("nums3").versions) == 1

    # check checkpoints
    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    # -------------- FIRST RUN -------------------
    first_job_id = catalog.metastore.create_job("my-job", "echo 1;")
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).save("nums2")
    dc.read_dataset("nums", session=test_session).save("nums3")

    # -------------- SECOND RUN -------------------
    second_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=first_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).filter(dc.C("num") > 1).save(
        "nums2"
    )  # added change from first run
    dc.read_dataset("nums", session=test_session).save("nums3")

    # check that we skipped creation of first 2 datasets (they have only 1 version)
    assert (
        len(catalog.get_dataset("nums1").versions) == 1 if not reset_checkpoints else 2
    )
    assert len(catalog.get_dataset("nums2").versions) == 2
    assert len(catalog.get_dataset("nums3").versions) == 2

    # check checkpoints
    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    # -------------- FIRST RUN -------------------
    first_job_id = catalog.metastore.create_job("my-job", "echo 1;")
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).save("nums2")
    with pytest.raises(DataChainError):
        (
            dc.read_dataset("nums", session=test_session)
            .map(new=mapper_fail)
            .save("nums3")
        )
    # check we created first 2 datasets but third one failed
    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    second_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=first_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).save("nums2")
    dc.read_dataset("nums", session=test_session).save("nums3")

    # -------------- THIRD RUN -------------------
    third_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=second_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", third_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).filter(dc.C("num") > 1).save(
        "nums2"
    )  # added change from first run
    with pytest.raises(DataChainError):
        (
            dc.read_dataset("nums", session=test_session)
            .map(new=mapper_fail)
            .save("nums3")
        )

    # -------------- FOURTH RUN -------------------
    fourth_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=third_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", fourth_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")
    dc.read_dataset("nums", session=test_session).filter(dc.C("num") > 1).save(
        "nums2"
    )  # added change from first run
    dc.read_dataset("nums", session=test_session).save("nums3")

    num1_versions = len(catalog.get_dataset("nums1").versions)
    num2_versions = len(catalog.get_dataset("nums2").versions)
    num3_versions = len(catalog.get_dataset("nums3").versions)

    if reset_checkpoints:
        assert num1_versions == 4
        assert num2_versions == 4
        assert num3_versions == 2

    else:
        assert num1_versions == 1
        assert num2_versions == 2
        assert num3_versions == 2

    # check checkpoints
    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(fourth_job_id))) == 3


def test_checkpoints_check_valid_chain_is_returned(
    test_session,
    monkeypatch,
    nums_dataset,
):
    catalog = test_session.catalog

    # -------------- FIRST RUN -------------------
    first_job_id = catalog.metastore.create_job("my-job", "echo 1;")
    monkeypatch.setenv("DATACHAIN_JOB_ID", first_job_id)
    dc.read_dataset("nums", session=test_session).save("nums1")

    # -------------- SECOND RUN -------------------
    second_job_id = catalog.metastore.create_job(
        "my-job", "echo 1;", parent_job_id=first_job_id
    )
    monkeypatch.setenv("DATACHAIN_JOB_ID", second_job_id)
    ds = dc.read_dataset("nums", session=test_session).save("nums1")

    # checking that we return expected DataChain even though we skipped chain creation
    # because of the checkpoints
    assert ds.dataset.name == "nums1"
    assert len(ds.dataset.versions) == 1
    assert ds.order_by("num").to_list("num") == [(1,), (2,), (3,)]
