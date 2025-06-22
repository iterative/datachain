from typing import Optional

from datachain.error import NamespaceCreateNotAllowedError
from datachain.namespace import Namespace
from datachain.query import Session


def create(
    name: str, description: Optional[str] = None, session: Optional[Session] = None
) -> Namespace:
    """
    Creates a new custom namespace.
    A Namespace is an object used to organize datasets. It has name and a list of
    Project objects underneath it. On the other hand, each Project can have multiple
    datasets.
    Note that creating namespaces is not allowed in the local environment, unlike
    in Studio, where it is allowed.
    In local environment all datasets are created under the default `local` namespace.

    Parameters:
        name : The name of the namespace.
        description : A description of the namespace.
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

    Namespace.validate_name(name)

    return session.catalog.metastore.create_namespace(name, description)


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
        import datachain as dc
        namespaces = dc.namespaces.ls()
        ```
    """
    return Session.get(session).catalog.metastore.list_namespaces()
