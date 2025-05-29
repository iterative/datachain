import pytest

import datachain as dc
from datachain.error import (
    NamespaceNotFoundError,
    ProjectCreateNotAllowedError,
    ProjectNotFoundError,
)
from datachain.namespace import Namespace
from datachain.project import Project
from tests.utils import skip_if_not_sqlite


@pytest.fixture
def dev_namespace(test_session):
    return dc.namespaces.create("dev", "Dev namespace")


@pytest.fixture
def chatbot_project(test_session, dev_namespace):
    return dc.projects.create("chatbot", "dev", "Chatbot project")


def test_create_project(test_session, dev_namespace):
    project = dc.projects.create("chatbot", dev_namespace.name, session=test_session)
    assert project.id
    assert project.uuid
    assert project.created_at
    assert project.name == "chatbot"
    assert project.namespace == dev_namespace


def test_create_project_namespace_does_not_exist(test_session):
    with pytest.raises(NamespaceNotFoundError) as excinfo:
        dc.projects.create("chatbot", "wrong", session=test_session)

    assert str(excinfo.value) == "Namespace wrong not found."


def test_create_project_that_already_exists_in_namespace(test_session, dev_namespace):
    name = "chatbot"
    dc.projects.create(name, dev_namespace.name, "desc 1", session=test_session)
    project = dc.projects.create(
        name, dev_namespace.name, "desc 2", session=test_session
    )
    assert project.description == "desc 1"


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
    assert dev_project.description == "Dev chatbot"
    assert prod_project.name == name
    assert prod_project.description == "Prod chatbot"


def test_create_with_reserved_name(test_session, dev_namespace):
    with pytest.raises(ValueError) as excinfo:
        dc.projects.create("local", dev_namespace.name, session=test_session)

    assert str(excinfo.value) == "Project name local is reserved."


@pytest.mark.disable_autouse
def test_create_by_user_not_allowed(test_session):
    with pytest.raises(ProjectCreateNotAllowedError) as excinfo:
        dc.projects.create("chatbot", "dev", session=test_session)

    assert str(excinfo.value) == "Creating custom project is not allowed"


def test_get_project(test_session, chatbot_project, dev_namespace):
    project = dc.projects.get(
        chatbot_project.name, dev_namespace.name, session=test_session
    )
    assert project.id == chatbot_project.id
    assert project.uuid == chatbot_project.uuid
    assert project.name == chatbot_project.name
    assert project.description == chatbot_project.description
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
    local_project = dc.projects.get(
        Project.default(), Namespace.default(), session=test_session
    )
    assert local_project.name == Project.default()
