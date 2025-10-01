import pytest

import datachain as dc
from datachain.error import DatasetNotFoundError, JobNotFoundError
from datachain.job import job_manager
from datachain.lib.utils import DataChainError


def mapper_fail(num) -> int:
    raise Exception("Error")


@pytest.fixture
def nums_dataset(test_session):
    return dc.read_values(num=[1, 2, 3], session=test_session).save("nums")


@pytest.fixture
def reset_job_manager():
    job_manager.reset()


@pytest.mark.parametrize("reset_checkpoints", [True, False])
@pytest.mark.parametrize("with_delta", [True, False])
@pytest.mark.parametrize("use_datachain_job_id_env", [True, False])
def test_checkpoints(
    test_session,
    monkeypatch,
    nums_dataset,
    reset_checkpoints,
    with_delta,
    use_datachain_job_id_env,
):
    catalog = test_session.catalog
    metastore = catalog.metastore

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    if with_delta:
        chain = dc.read_dataset(
            "nums", delta=True, delta_on=["num"], session=test_session
        )
    else:
        chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    job_manager.reset()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID", metastore.create_job("my-job", "echo 1;")
        )

    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(DataChainError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = job_manager.job.id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    job_manager.reset()
    if use_datachain_job_id_env:
        monkeypatch.setenv(
            "DATACHAIN_JOB_ID",
            metastore.create_job("my-job", "echo 1;", parent_job_id=first_job_id),
        )

    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = job_manager.job.id

    assert len(catalog.get_dataset("nums1").versions) == 2 if reset_checkpoints else 1
    assert len(catalog.get_dataset("nums2").versions) == 2 if reset_checkpoints else 1
    assert len(catalog.get_dataset("nums3").versions) == 1

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_modified_chains(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    first_job_id = job_manager.job.id

    # -------------- SECOND RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")  # added change from first run
    chain.save("nums3")
    second_job_id = job_manager.job.id

    assert len(catalog.get_dataset("nums1").versions) == 2 if reset_checkpoints else 1
    assert len(catalog.get_dataset("nums2").versions) == 2
    assert len(catalog.get_dataset("nums3").versions) == 2

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3


@pytest.mark.parametrize("reset_checkpoints", [True, False])
def test_checkpoints_multiple_runs(
    test_session, monkeypatch, nums_dataset, reset_checkpoints
):
    catalog = test_session.catalog

    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", reset_checkpoints)

    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.save("nums2")
    with pytest.raises(DataChainError):
        chain.map(new=mapper_fail).save("nums3")
    first_job_id = job_manager.job.id

    catalog.get_dataset("nums1")
    catalog.get_dataset("nums2")
    with pytest.raises(DatasetNotFoundError):
        catalog.get_dataset("nums3")

    # -------------- SECOND RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.save("nums2")
    chain.save("nums3")
    second_job_id = job_manager.job.id

    # -------------- THIRD RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    with pytest.raises(DataChainError):
        chain.map(new=mapper_fail).save("nums3")
    third_job_id = job_manager.job.id

    # -------------- FOURTH RUN -------------------
    job_manager.reset()
    chain.save("nums1")
    chain.filter(dc.C("num") > 1).save("nums2")
    chain.save("nums3")
    fourth_job_id = job_manager.job.id

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

    assert len(list(catalog.metastore.list_checkpoints(first_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(second_job_id))) == 3
    assert len(list(catalog.metastore.list_checkpoints(third_job_id))) == 2
    assert len(list(catalog.metastore.list_checkpoints(fourth_job_id))) == 3


def test_checkpoints_check_valid_chain_is_returned(
    test_session,
    monkeypatch,
    nums_dataset,
):
    monkeypatch.setenv("DATACHAIN_CHECKPOINTS_RESET", False)
    chain = dc.read_dataset("nums", session=test_session)

    # -------------- FIRST RUN -------------------
    job_manager.reset()
    chain.save("nums1")

    # -------------- SECOND RUN -------------------
    job_manager.reset()
    ds = chain.save("nums1")

    # checking that we return expected DataChain even though we skipped chain creation
    # because of the checkpoints
    assert ds.dataset.name == "nums1"
    assert len(ds.dataset.versions) == 1
    assert ds.order_by("num").to_list("num") == [(1,), (2,), (3,)]


def test_checkpoints_invalid_parent_job_id(
    test_session, monkeypatch, nums_dataset, reset_job_manager
):
    # setting wrong job id
    monkeypatch.setenv("DATACHAIN_JOB_ID", "caee6c54-6328-4bcd-8ca6-2b31cb4fff94")
    with pytest.raises(JobNotFoundError):
        dc.read_dataset("nums", session=test_session).save("nums1")
