import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate a list of N random points in 2D space.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: List of N tuples, each containing two random floats between 0 and 1.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generate a split dataset where points are classified based on their first coordinate.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points and their classifications.
                Points with x_1 < 0.5 are classified as 1, others as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generate a split dataset where points are classified based on their x-coordinate.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points and their classifications.
               Points with x < 0.2 or x > 0.8 are classified as 1, others as 0.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generate a split dataset with a specified number of points.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points and their classifications.
               [Additional details about how points are split/classified would go here]

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generate a dataset with points classified based on an XOR-like condition.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points and their classifications.
               Points are classified as 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5),
               otherwise they are classified as 0.

    Notes:
    -----
    The function uses `make_pts(N)` to generate the initial set of points.
    The classification creates an XOR-like pattern in the 2D space.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generate a dataset with points classified based on their position relative to a circle.

    Args:
    ----
        N (int): Number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing N points and their classifications.
               Points are likely classified based on whether they fall inside or outside
               a circular boundary in the 2D space.

    Notes:
    -----
    The exact classification criteria and circle parameters are not specified in the
    function signature. This docstring assumes a circular classification based on the
    function name.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generate a dataset of points arranged in two intertwined spirals.

    Args:
    ----
        N (int): Total number of points to generate. Should be even as it's split equally
                 between two spirals.

    Returns:
    -------
        Graph: A Graph object containing N points arranged in two spirals and their classifications.
               The first N//2 points form one spiral and are classified as 0.
               The second N//2 points form another spiral and are classified as 1.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
