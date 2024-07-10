import math

import numpy as np


def euclidean_distance(a: str, b: str):
    a_np = np.fromstring(a[1:-1], sep=",")
    b_np = np.fromstring(b[1:-1], sep=",")

    return np.linalg.norm(b_np - a_np)


def cosine_distance(a: str, b: str):
    u = np.fromstring(a[1:-1], sep=",")
    v = np.fromstring(b[1:-1], sep=",")

    uv = np.inner(u, v)
    uu = np.inner(u, u)
    vv = np.inner(v, v)

    dist = 1.0 - uv / math.sqrt(uu * vv)

    return max(0, min(dist, 2.0))
