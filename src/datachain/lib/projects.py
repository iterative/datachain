from typing import Optional

from datachain.error import ProjectCreateNotAllowedError
from datachain.lib.namespaces import get as get_namespace
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
        project = dc.create_project("my-project", "dev", "My personal project")
        ```
    """
    if not Project.allowed_to_create():
        raise ProjectCreateNotAllowedError("Creating custom project is not allowed")
    if name in Project.reserved_names():
        raise ValueError(f"Project name {name} is reserved.")

    session = Session.get(session)
    namespace = get_namespace(namespace_name, session=session)
    return session.catalog.metastore.create_project(name, namespace, description)


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
        namespace = dc.get_project("my-project", "local")
        ```
    """
    session = Session.get(session)
    namespace = get_namespace(namespace_name, session=session)
    return session.catalog.metastore.get_project(name, namespace)
