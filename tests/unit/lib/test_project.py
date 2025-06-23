import pytest

import datachain as dc
from datachain.error import (
    InvalidProjectNameError,
    ProjectCreateNotAllowedError,
    ProjectNotFoundError,
)
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def dev_namespace(test_session):
    return dc.namespaces.create("dev", "Dev namespace")


@pytest.fixture
def chatbot_project(test_session, dev_namespace):
    return dc.projects.create("chatbot", "dev", "Chatbot project")


@pytest.mark.parametrize("namespace_created_upfront", (True, False))
def test_create_project(test_session, namespace_created_upfront):
    if namespace_created_upfront:
        dc.namespaces.create("dev")

    project = dc.projects.create("chatbot", "dev", session=test_session)
    assert project.id
    assert project.uuid
    assert project.created_at
    assert project.name == "chatbot"
    assert project.namespace == dc.namespaces.get("dev", session=test_session)


def test_create_project_that_already_exists_in_namespace(test_session, dev_namespace):
    name = "chatbot"
    dc.projects.create(name, dev_namespace.name, "desc 1", session=test_session)
    project = dc.projects.create(
        name, dev_namespace.name, "desc 2", session=test_session
    )
    assert project.descr == "desc 1"


def test_create_project_with_the_same_name_in_different_namespace(test_session):
    name = "chatbot"
    dev_namespace = dc.namespaces.create("dev")
    prod_namespace = dc.namespaces.create("prod")

    dev_project = dc.projects.create(
        name, dev_namespace.name, "Dev chatbot", session=test_session
    )
    prod_project = dc.projects.create(
        name, prod_namespace.name, "Prod chatbot", session=test_session
    )

    assert dev_project.name == name
    assert dev_project.descr == "Dev chatbot"
    assert prod_project.name == name
    assert prod_project.descr == "Prod chatbot"


@pytest.mark.parametrize("name", ["local", "with.dots", ""])
def test_invalid_name(test_session, dev_namespace, name):
    with pytest.raises(InvalidProjectNameError):
        dc.projects.create(name, dev_namespace.name, session=test_session)


@pytest.mark.disable_autouse
@skip_if_not_sqlite
def test_create_by_user_not_allowed(test_session):
    with pytest.raises(ProjectCreateNotAllowedError) as excinfo:
        dc.projects.create("chatbot", "dev", session=test_session)

    assert str(excinfo.value) == "Creating project is not allowed"


def test_get_project(test_session, chatbot_project, dev_namespace):
    project = dc.projects.get(
        chatbot_project.name, dev_namespace.name, session=test_session
    )
    assert project.id == chatbot_project.id
    assert project.uuid == chatbot_project.uuid
    assert project.name == chatbot_project.name
    assert project.descr == chatbot_project.descr
    assert project.created_at == chatbot_project.created_at
    assert project.namespace == dev_namespace


def test_get_project_not_found(test_session, dev_namespace):
    with pytest.raises(ProjectNotFoundError) as excinfo:
        dc.projects.get("wrong", dev_namespace.name, session=test_session)

    assert str(excinfo.value) == "Project wrong in namespace dev not found."


def test_get_project_not_found_but_exists_in_other_namespace(
    test_session, dev_namespace, chatbot_project
):
    name = "images"
    prod_namespace = dc.namespaces.create("prod")
    dc.projects.create(name, prod_namespace.name, session=test_session)

    dc.projects.get(name, prod_namespace.name, session=test_session)
    with pytest.raises(ProjectNotFoundError) as excinfo:
        dc.projects.get(name, dev_namespace.name, session=test_session)

    assert str(excinfo.value) == f"Project {name} in namespace dev not found."


@skip_if_not_sqlite
def test_local_project_is_created(test_session):
    project_class = test_session.catalog.metastore.project_class
    namespace_class = test_session.catalog.metastore.namespace_class
    local_project = dc.projects.get(
        project_class.default(), namespace_class.default(), session=test_session
    )
    assert local_project.name == project_class.default()


def test_ls_projects(test_session):
    metastore = test_session.catalog.metastore

    ns1 = dc.namespaces.create("ns1")
    ns2 = dc.namespaces.create("ns2")

    p_names = ["p1", "p2", "p3"]
    for name in p_names:
        dc.projects.create(name, ns1.name, "", session=test_session)
        dc.projects.create(name, ns2.name, "", session=test_session)

    projects = dc.projects.ls(session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            (metastore.default_namespace_name, metastore.default_project_name),
            (metastore.system_namespace_name, metastore.listing_project_name),
            ("ns1", "p1"),
            ("ns1", "p2"),
            ("ns1", "p3"),
            ("ns2", "p1"),
            ("ns2", "p2"),
            ("ns2", "p3"),
        ]
    )


def test_ls_projects_one_namespace(test_session):
    ns1 = dc.namespaces.create("ns1")
    ns2 = dc.namespaces.create("ns2")

    p_names = ["p1", "p2", "p3"]
    for name in p_names:
        dc.projects.create(name, ns1.name, "", session=test_session)
        dc.projects.create(name, ns2.name, "", session=test_session)

    projects = dc.projects.ls("ns1", session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            ("ns1", "p1"),
            ("ns1", "p2"),
            ("ns1", "p3"),
        ]
    )


def test_ls_projects_just_default(test_session):
    metastore = test_session.catalog.metastore

    projects = dc.projects.ls(session=test_session)
    assert sorted([(p.namespace.name, p.name) for p in projects]) == sorted(
        [
            (metastore.default_namespace_name, metastore.default_project_name),
            (metastore.system_namespace_name, metastore.listing_project_name),
        ]
    )


def test_ls_projects_empty_in_namespace(test_session):
    dc.namespaces.create("ns1")
    projects = dc.projects.ls("ns1", session=test_session)
    assert [(p.namespace.name, p.name) for p in projects] == []
