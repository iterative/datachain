"""
Functional tests for read_dataset with PEP 440 version specifiers.
"""

import pytest

import datachain as dc
from datachain.error import DatasetVersionNotFoundError


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_read_dataset_version_specifiers(cloud_test_catalog):
    """Test read_dataset with various PEP 440 version specifiers."""
    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    # Create a dataset with multiple versions
    dataset_name = "test_version_specifiers"

    # Version 1.0.0
    (
        dc.read_storage(f"{src_uri}/dogs", session=session)
        .limit(2)
        .save(dataset_name, version="1.0.0")
    )

    # Version 1.1.0
    (
        dc.read_storage(f"{src_uri}/dogs", session=session)
        .limit(3)
        .save(dataset_name, version="1.1.0")
    )

    # Version 1.2.0
    (
        dc.read_storage(f"{src_uri}/dogs", session=session)
        .limit(4)
        .save(dataset_name, version="1.2.0")
    )

    # Version 2.0.0
    (
        dc.read_storage(f"{src_uri}/cats", session=session)
        .limit(2)
        .save(dataset_name, version="2.0.0")
    )

    # Test exact version specifier
    result = dc.read_dataset(dataset_name, version="==1.1.0", session=session)
    assert result.count() == 3

    # Test greater than or equal specifier - should get latest (2.0.0)
    result = dc.read_dataset(dataset_name, version=">=1.1.0", session=session)
    assert result.count() == 2  # version 2.0.0 has 2 items

    result = dc.read_dataset(dataset_name, version="1.2.0", session=session)
    assert result.count() == 4  # version 1.2.0 has 4 items

    # Test less than specifier - should get 1.2.0 (latest before 2.0.0)
    result = dc.read_dataset(dataset_name, version="<2.0.0", session=session)
    assert result.count() == 4  # version 1.2.0 has 4 items

    # Test compatible release specifier - should get latest 1.x (1.2.0)
    result = dc.read_dataset(dataset_name, version="~=1.0", session=session)
    assert result.count() == 4  # version 1.2.0 has 4 items

    # Test version pattern - should get latest 1.x (1.2.0)
    result = dc.read_dataset(dataset_name, version="==1.*", session=session)
    assert result.count() == 4  # version 1.2.0 has 4 items

    # Test complex specifier - should get 1.2.0
    result = dc.read_dataset(dataset_name, version=">=1.1.0,<2.0.0", session=session)
    assert result.count() == 4  # version 1.2.0 has 4 items


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_read_dataset_version_specifiers_no_match(cloud_test_catalog):
    """Test read_dataset with version specifiers that don't match any version."""
    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    # Create a dataset with a single version
    dataset_name = "test_no_match_specifiers"

    (
        dc.read_storage(f"{src_uri}/dogs", session=session)
        .limit(2)
        .save(dataset_name, version="1.0.0")
    )

    # Test version specifier that doesn't match any existing version
    with pytest.raises(DatasetVersionNotFoundError) as exc_info:
        dc.read_dataset(dataset_name, version=">=2.0.0", session=session)

    assert (
        "No dataset test_no_match_specifiers version matching specifier >=2.0.0"
        in str(exc_info.value)
    )


@pytest.mark.parametrize("cloud_type", ["file"], indirect=True)
def test_read_dataset_version_specifiers_exact_version(cloud_test_catalog):
    """Test that version specifiers work alongside with exact version reads."""
    session = cloud_test_catalog.session
    src_uri = cloud_test_catalog.src_uri

    # Create a dataset with multiple versions
    dataset_name = "test_backward_compatibility"

    (
        dc.read_storage(f"{src_uri}/dogs", session=session)
        .limit(2)
        .save(dataset_name, version="1.0.0")
    )

    # Test reading by exact version
    result = dc.read_dataset(dataset_name, version="1.0.0", session=session)
    assert result.count() == 2  # exact version 1.0.0 has 2 items

    # Test reading by exact version int - backward compatibility
    result = dc.read_dataset(dataset_name, version=1, session=session)
    assert result.count() == 2  # exact version 1.0.0 has 2 items
