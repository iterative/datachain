from typing import Optional

from datachain.error import ProjectCreateNotAllowedError
from datachain.project import Project
from datachain.query import Session


def create(
    name: str,
    namespace: str,
    descr: Optional[str] = None,
    session: Optional[Session] = None,
) -> Project:
    """
    Creates a new project.
    A Project is an object used to organize datasets. It is created under a
    specific namespace and can have multiple datasets.
    Default project is always automatically created and is used if not explicitly
    specified otherwise.
    In Studio user can create multiple projects, while in CLI only default project
    can be used.

    Parameters:
        name : The name of the project.
        namespace : The name of the namespace under which the new project is being
            created. If namespace doesn't exist, it will be created automatically.
        descr : A description of the project.
        session : Session to use for creating project.

    Example:
        ```py
        import datachain as dc
        project = dc.projects.create("my-project", "dev", "My personal project")
        ```
    """
    session = Session.get(session)

    if not session.catalog.metastore.project_allowed_to_create:
        raise ProjectCreateNotAllowedError("Creating project is not allowed")

    Project.validate_name(name)

    return session.catalog.metastore.create_project(name, namespace, descr)


def get(name: str, namespace: str, session: Optional[Session]) -> Project:
    """
    Gets a project by name in some namespace.
    If the project is not found, a `ProjectNotFoundError` is raised.

    Parameters:
        name : The name of the project.
        namespace : The name of the namespace.
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        project  = dc.get_project("my-project", "local")
        ```
    """
    return Session.get(session).catalog.metastore.get_project(name, namespace)


def ls(
    namespace: Optional[str] = None, session: Optional[Session] = None
) -> list[Project]:
    """
    Gets a list of projects in a specific namespace or from all namespaces.

    Parameters:
        namespace : An optional namespace name.
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
    if namespace:
        namespace_id = session.catalog.metastore.get_namespace(namespace).id

    return session.catalog.metastore.list_projects(namespace_id)
