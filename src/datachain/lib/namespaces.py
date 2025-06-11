from typing import Optional

from datachain.error import NamespaceCreateNotAllowedError
from datachain.namespace import Namespace
from datachain.query import Session


def create(
    name: str, description: Optional[str] = None, session: Optional[Session] = None
) -> Namespace:
    """
    Creates new custom namespace.
    Note that creating namespaces is not allowed for local environment, unlike in
    Studio where it is allowed.
    In local environment all datasets are created under default `local` namespace.

    Parameters:
        name : namespace name.
        description : namespace description.
        session : Session to use for creating namespace.

    Example:
        ```py
        import datachain as dc
        namespace = dc.namespaces.create("dev", "Dev namespace")
        ```
    """
    session = Session.get(session)

    if not session.catalog.metastore.namespace_allowed_to_create:
        raise NamespaceCreateNotAllowedError("Creating custom namespace is not allowed")
    if name in Namespace.reserved_names():
        raise ValueError(f"Namespace name {name} is reserved.")
    return session.catalog.metastore.create_namespace(name, description)


def get(name: str, session: Optional[Session]) -> Namespace:
    """
    Gets namespace by name.
    If namespace is not found, `NamespaceNotFoundError` is thrown.

    Parameters:
        name : namespace name.
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
    Gets list of namespaces.

    Parameters:
        session : Session to use for getting project.

    Example:
        ```py
        import datachain as dc
        namespaces = dc.namespaces.ls()
        ```
    """
    return Session.get(session).catalog.metastore.list_namespaces()
