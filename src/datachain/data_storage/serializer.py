import base64
import pickle
from abc import abstractmethod
from collections.abc import Callable
from typing import Any


class Serializable:
    @abstractmethod
    def clone_params(self) -> tuple[Callable[..., Any], list[Any], dict[str, Any]]:
        """
        Returns the class, args, and kwargs needed to instantiate a cloned copy
        of this instance for use in separate processes or machines.
        """

    def serialize(self) -> str:
        """
        Returns a string representation of clone params.
        This is useful for storing the state of an object in environment variable.
        """
        return base64.b64encode(pickle.dumps(self.clone_params())).decode()


def deserialize(s: str) -> Serializable:
    """
    Returns a new instance of the class represented by the string.
    """
    (f, args, kwargs) = pickle.loads(base64.b64decode(s.encode()))  # noqa: S301
    return f(*args, **kwargs)
