import pytest

from datachain.error import (
    InvalidNamespaceNameError,
    NamespaceCreateNotAllowedError,
    NamespaceNotFoundError,
)
from datachain.lib.namespaces import create as create_namespace
from datachain.lib.namespaces import get as get_namespace
from datachain.lib.namespaces import ls as ls_namespaces
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def dev_namespace(test_session):
    return create_namespace("dev", "Dev namespace")


def test_create_namespace(test_session):
    namespace = create_namespace("dev", session=test_session)
    assert namespace.name
    assert namespace.id
    assert namespace.uuid
    assert namespace.created_at


@pytest.mark.parametrize("allow_create_namespace", [False])
@skip_if_not_sqlite
def test_create_by_user_not_allowed(test_session, allow_create_namespace):
    with pytest.raises(NamespaceCreateNotAllowedError) as excinfo:
        create_namespace("dev", session=test_session)

    assert str(excinfo.value) == "Creating namespace is not allowed"


def test_create_namespace_already_exists(test_session):
    namespace1 = create_namespace("dev", session=test_session)
    namespace2 = create_namespace("dev", session=test_session)
    assert namespace1.id == namespace2.id


@pytest.mark.parametrize("name", ["local", "with.dots", ""])
def test_invalid_name(test_session, name):
    with pytest.raises(InvalidNamespaceNameError):
        create_namespace(name, session=test_session)


def test_get_namespace(test_session, dev_namespace):
    namespace = get_namespace("dev", session=test_session)
    assert namespace.id == dev_namespace.id
    assert namespace.uuid == dev_namespace.uuid
    assert namespace.name == dev_namespace.name
    assert namespace.descr == dev_namespace.descr
    assert namespace.created_at == dev_namespace.created_at


def test_get_namespace_not_found(test_session, dev_namespace):
    with pytest.raises(NamespaceNotFoundError) as excinfo:
        get_namespace("wrong", session=test_session)

    assert str(excinfo.value) == "Namespace wrong not found."


def test_ls_namespaces(test_session):
    system_namespace_name = test_session.catalog.metastore.system_namespace_name

    create_namespace("ns1")
    create_namespace("ns2")

    namespaces = ls_namespaces(session=test_session)
    assert sorted([n.name for n in namespaces]) == sorted(
        [system_namespace_name, "ns1", "ns2"]
    )


def test_ls_namespaces_just_local(test_session):
    system_namespace_name = test_session.catalog.metastore.system_namespace_name
    namespaces = ls_namespaces(session=test_session)
    assert [n.name for n in namespaces] == [system_namespace_name]
