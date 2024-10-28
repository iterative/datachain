from .func import Func
from .inner import random


def rand() -> Func:
    """
    Returns the random integer value.

    Returns:
        Func: A Func object that represents the rand function.

    Example:
        ```py
        dc.mutate(
            rnd=func.random.rand(),
        )
        ```

    Note:
        - Result column will always be of type integer.
    """
    return Func("rand", inner=random.rand, result_type=int)
