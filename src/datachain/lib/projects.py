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
    Creates new custom project.
    Note that creating projects is not allowed for local environment, unlike in
    Studio where it is allowed.
    In local environment all datasets are created under default `local` project.
    Project is also connected to it's parent namespace.

    Parameters:
        name : project name.
        namespace : namespace name under which we are creating new project.
        description : project description.
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
    Gets project by name in some namespace.
    If project is not found, `ProjectNotFoundError` is thrown.

    Parameters:
        name : project name.
        namespace_name : namespace name.
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
    Gets list of projects in some namespace or in general (all namespaces).

    Parameters:
        namespace_name : optional namespace name.
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
