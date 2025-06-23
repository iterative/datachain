from typing import Optional

from datachain.error import NamespaceCreateNotAllowedError
from datachain.namespace import Namespace
from datachain.query import Session


def create(
    name: str, descr: Optional[str] = None, session: Optional[Session] = None
) -> Namespace:
    """
    Creates a new namespace.
    A Namespace is an object used to organize datasets. It can have multiple projects
    and each project can have multiple datasets.
    Default namespace is always automatically created and is used if not explicitly
    specified otherwise.
    In Studio user can create multiple namespaces, while in CLI only default namespace
    can be used.

    Parameters:
        name : The name of the namespace.
        descr : A description of the namespace.
        session : Session to use for creating namespace.

    Example:
        ```py
        from datachain.lib.namespaces import create as create_namespace
        namespace = create_namespace("dev", "Dev namespace")
        ```
    """
    session = Session.get(session)

    if not session.catalog.metastore.namespace_allowed_to_create:
        raise NamespaceCreateNotAllowedError("Creating namespace is not allowed")

    Namespace.validate_name(name)

    return session.catalog.metastore.create_namespace(name, descr)


def get(name: str, session: Optional[Session] = None) -> Namespace:
    """
    Gets a namespace by name.
    If the namespace is not found, a `NamespaceNotFoundError` is raised.

    Parameters:
        name : The name of the namespace.
        session : Session to use for getting namespace.

    Example:
        ```py
        import datachain as dc
        namespace = dc.get_namespace("local")
        ```
    """
    session = Session.get(session)
    return session.catalog.metastore.get_namespace(name)


def ls(session: Optional[Session] = None) -> list[Namespace]:
    """
    Gets a list of all namespaces.

    Parameters:
        session : Session to use for getting namespaces.

    Example:
        ```py
        from datachain.lib.namespaces import ls as ls_namespaces
        namespaces = ls_namespaces()
        ```
    """
    return Session.get(session).catalog.metastore.list_namespaces()
