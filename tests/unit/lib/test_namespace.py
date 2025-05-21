from unittest.mock import patch

import pytest

import datachain as dc
from datachain.error import NamespaceCreateNotAllowedError, NamespaceNotFoundError
from datachain.namespace import Namespace


@pytest.fixture
def mock_allowed_to_create_namespace():
    with patch("datachain.namespaces.Namespace", wraps=Namespace) as mock_namespace:
        mock_namespace.allowed_to_create.return_value = True
        yield mock_namespace


@pytest.fixture
def dev_namespace(test_session, mock_allowed_to_create_namespace):
    return dc.namespaces.create("dev", "Dev namespace")


def test_create_namespace(test_session, mock_allowed_to_create_namespace):
    namespace = dc.namespaces.create("dev", session=test_session)
    assert namespace.name
    assert namespace.id
    assert namespace.uuid
    assert namespace.created_at


def test_create_by_user_not_allowed(test_session):
    with pytest.raises(NamespaceCreateNotAllowedError) as excinfo:
        dc.namespaces.create("dev", session=test_session)

    assert str(excinfo.value) == "Creating custom namespace is not allowed"


def test_create_namespace_already_exists(
    test_session, mock_allowed_to_create_namespace
):
    namespace1 = dc.namespaces.create("dev", session=test_session)
    namespace2 = dc.namespaces.create("dev", session=test_session)
    assert namespace1.id == namespace2.id


def test_create_with_reserved_name(test_session, mock_allowed_to_create_namespace):
    with pytest.raises(ValueError) as excinfo:
        dc.namespaces.create("local", session=test_session)

    assert str(excinfo.value) == "Namespace name local is reserved."


def test_get_namespace(test_session, dev_namespace):
    namespace = dc.namespaces.get("dev", session=test_session)
    assert namespace.id == dev_namespace.id
    assert namespace.uuid == dev_namespace.uuid
    assert namespace.name == dev_namespace.name
    assert namespace.description == dev_namespace.description
    assert namespace.created_at == dev_namespace.created_at


def test_get_namespace_not_found(test_session, dev_namespace):
    with pytest.raises(NamespaceNotFoundError) as excinfo:
        dc.namespaces.get("wrong", session=test_session)

    assert str(excinfo.value) == "Namespace wrong not found."
