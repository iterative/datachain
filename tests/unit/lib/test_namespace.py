import pytest

import datachain as dc
from datachain.error import NamespaceCreateNotAllowedError, NamespaceNotFoundError


@pytest.fixture
def dev_namespace(test_session):
    return dc.namespaces.create("dev", "Dev namespace")


def test_create_namespace(test_session):
    namespace = dc.namespaces.create("dev", session=test_session)
    assert namespace.name
    assert namespace.id
    assert namespace.uuid
    assert namespace.created_at


@pytest.mark.disable_autouse
def test_create_by_user_not_allowed(test_session):
    with pytest.raises(NamespaceCreateNotAllowedError) as excinfo:
        dc.namespaces.create("dev", session=test_session)

    assert str(excinfo.value) == "Creating custom namespace is not allowed"


def test_create_namespace_already_exists(test_session):
    namespace1 = dc.namespaces.create("dev", session=test_session)
    namespace2 = dc.namespaces.create("dev", session=test_session)
    assert namespace1.id == namespace2.id


def test_create_with_reserved_name(test_session):
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


def test_local_namespace_is_created(test_session):
    namespace_class = test_session.catalog.metastore.namespace_class
    local_namespace = dc.namespaces.get(namespace_class.default(), session=test_session)
    assert local_namespace.name == namespace_class.default()
