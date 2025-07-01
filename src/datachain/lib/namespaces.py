from typing import Optional

from datachain.error import NamespaceCreateNotAllowedError
from datachain.namespace import Namespace
from datachain.query import Session


def create(
    name: str, descr: Optional[str] = None, session: Optional[Session] = None
) -> Namespace:
    """
    Creates a new namespace.

    Namespaces organize projects, which in turn organize datasets. A default
    namespace always exists and is used if none is specified. Multiple namespaces
    can be created in Studio, but only the default is available in the CLI.

    Parameters:
        name: Name of the new namespace.
        descr: Optional description of the namespace.
        session: Optional session to use for the operation.

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
