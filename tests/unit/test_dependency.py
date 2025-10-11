import datetime

import pytest

from datachain.catalog.catalog import _populate_dependency_tree, extract_flat_ids
from datachain.dataset import DatasetDependency


@pytest.fixture
def dataset_deps():
    return {
        "249": DatasetDependency(
            id=249,
            type="storage",
            namespace="system",
            project="listing",
            name="lst__gs://amrit-datachain-test/",
            version="1.0.0",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 20, 39, 474881, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "250": DatasetDependency(
            id=250,
            type="storage",
            namespace="system",
            project="listing",
            name="lst__gs://amrit-datachain-test/",
            version="1.0.0",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 20, 39, 474881, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "251": DatasetDependency(
            id=251,
            type="storage",
            namespace="system",
            project="listing",
            name="lst__az://amrit-test-az/",
            version="1.0.0",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 20, 41, 190267, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "252": DatasetDependency(
            id=252,
            type="storage",
            namespace="system",
            project="listing",
            name="lst__az://amrit-test-az/",
            version="1.0.0",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 20, 41, 190267, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "255": DatasetDependency(
            id=255,
            type="dataset",
            namespace="local",
            project="local",
            name="gs_pngs",
            version="1.0.3",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 24, 57, 108451, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "256": DatasetDependency(
            id=256,
            type="dataset",
            namespace="local",
            project="local",
            name="az_pngs",
            version="1.0.3",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 24, 57, 130564, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "297": DatasetDependency(
            id=297,
            type="dataset",
            namespace="local",
            project="local",
            name="all_pngs",
            version="1.0.3",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 24, 57, 150855, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
        "306": DatasetDependency(
            id=306,
            type="dataset",
            namespace="local",
            project="local",
            name="final_storage",
            version="1.0.4",
            created_at=datetime.datetime(
                2025, 10, 6, 11, 24, 57, 329456, tzinfo=datetime.timezone.utc
            ),
            dependencies=[],
        ),
    }


@pytest.fixture
def dependency_structure():
    return {"306": {"297": {"255": {"250": {}}, "256": {"252": {}}}}}


def test_populate_dependency_tree(dataset_deps, dependency_structure):
    result = _populate_dependency_tree(dataset_deps, dependency_structure)

    assert len(result) == 1  # Only one root dependency
    assert result[0].id == 306
    assert len(result[0].dependencies) == 1  # Only one dependency for the root
    assert result[0].dependencies[0].id == 297
    # Only two dependencies for first dependency
    assert len(result[0].dependencies[0].dependencies) == 2
    assert [d.id for d in result[0].dependencies[0].dependencies] == [255, 256]


def test_get_all_ids(dataset_deps):
    result = extract_flat_ids(dataset_deps)
    assert sorted(result) == sorted([249, 250, 251, 252, 255, 256, 297, 306])
