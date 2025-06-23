import pytest

import datachain as dc
from datachain.error import (
    InvalidNamespaceNameError,
    NamespaceCreateNotAllowedError,
    NamespaceNotFoundError,
)
from tests.utils import skip_if_not_sqlite


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
@skip_if_not_sqlite
def test_create_by_user_not_allowed(test_session):
    with pytest.raises(NamespaceCreateNotAllowedError) as excinfo:
        dc.namespaces.create("dev", session=test_session)

    assert str(excinfo.value) == "Creating namespace is not allowed"


def test_create_namespace_already_exists(test_session):
    namespace1 = dc.namespaces.create("dev", session=test_session)
    namespace2 = dc.namespaces.create("dev", session=test_session)
    assert namespace1.id == namespace2.id


@pytest.mark.parametrize("name", ["local", "with.dots", ""])
def test_invalid_name(test_session, name):
    with pytest.raises(InvalidNamespaceNameError):
        dc.namespaces.create(name, session=test_session)


def test_get_namespace(test_session, dev_namespace):
    namespace = dc.namespaces.get("dev", session=test_session)
    assert namespace.id == dev_namespace.id
    assert namespace.uuid == dev_namespace.uuid
    assert namespace.name == dev_namespace.name
    assert namespace.descr == dev_namespace.descr
    assert namespace.created_at == dev_namespace.created_at


def test_get_namespace_not_found(test_session, dev_namespace):
    with pytest.raises(NamespaceNotFoundError) as excinfo:
        dc.namespaces.get("wrong", session=test_session)

    assert str(excinfo.value) == "Namespace wrong not found."


def test_local_namespace_is_created(test_session):
    namespace_class = test_session.catalog.metastore.namespace_class
    local_namespace = dc.namespaces.get(namespace_class.default(), session=test_session)
    assert local_namespace.name == namespace_class.default()


def test_ls_namespaces(test_session):
    default_namespace_name = test_session.catalog.metastore.default_namespace_name
    system_namespace_name = test_session.catalog.metastore.system_namespace_name

    dc.namespaces.create("ns1")
    dc.namespaces.create("ns2")

    namespaces = dc.namespaces.ls(session=test_session)
    assert sorted([n.name for n in namespaces]) == sorted(
        [default_namespace_name, system_namespace_name, "ns1", "ns2"]
    )


def test_ls_namespaces_just_local(test_session):
    default_namespace_name = test_session.catalog.metastore.default_namespace_name
    system_namespace_name = test_session.catalog.metastore.system_namespace_name
    namespaces = dc.namespaces.ls(session=test_session)
    assert sorted([n.name for n in namespaces]) == sorted(
        [default_namespace_name, system_namespace_name]
    )
