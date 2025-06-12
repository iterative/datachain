from typing import Optional

from datachain.error import ProjectCreateNotAllowedError
from datachain.project import Project
from datachain.query import Session


def create(
    name: str,
    namespace_name: str,
    description: Optional[str] = None,
    session: Optional[Session] = None,
) -> Project:
    """
    Creates a new custom project.
    A Project is an object used to organize datasets. It is created under a
    specific namespace and has a list of datasets underneath it.
    Note that creating projects is not allowed in the local environment, unlike
    in Studio, where it is allowed.
    In local environment all datasets are created under the default `local` project.

    Parameters:
        name : The name of the project.
        namespace : The name of the namespace under which the new project is being
            created.
        description : A description of the project.
        session : Session to use for creating project.

    Example:
        ```py
        import datachain as dc
        project = dc.projects.create("my-project", "dev", "My personal project")
        ```
    """
    session = Session.get(session)

    if not session.catalog.metastore.project_allowed_to_create:
        raise ProjectCreateNotAllowedError("Creating custom project is not allowed")
    if name in Project.reserved_names():
        raise ValueError(f"Project name {name} is reserved.")

    return session.catalog.metastore.create_project(name, namespace_name, description)


def get(name: str, namespace_name: str, session: Optional[Session]) -> Project:
    """
    Gets a project by name in some namespace.
    If the project is not found, a `ProjectNotFoundError` is raised.

    Parameters:
        name : The name of the project.
        namespace_name : The name of the namespace.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        project  = dc.get_project("my-project", "local")
        ```
    """
    return Session.get(session).catalog.metastore.get_project(name, namespace_name)


def ls(
    namespace_name: Optional[str] = None, session: Optional[Session] = None
) -> list[Project]:
    """
    Gets a list of projects in a specific namespace or from all namespaces.

    Parameters:
        namespace_name : An optional namespace name.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        local_namespace_projects = dc.projects.ls("local")
        all_projects = dc.projects.ls()
        ```
    """
    session = Session.get(session)
    namespace_id = None
    if namespace_name:
        namespace_id = session.catalog.metastore.get_namespace(namespace_name).id

    return session.catalog.metastore.list_projects(namespace_id)
