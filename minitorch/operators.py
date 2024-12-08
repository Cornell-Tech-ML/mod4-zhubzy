"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Iterator, TypeVar, List


def add(a: float, b: float) -> float:
    """Adds two number"""
    return a + b


def eq(a: float, b: float) -> float:
    """Check if two numbers are equal"""
    return 1.0 if a == b else 0.0


def id(x: float) -> float:
    """Check if two numbers are equal"""
    return x


def exp(x: float) -> float:
    """Calculate the exponential of x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Check if two numbers are equal"""
    if x == 0:
        return 0
    return 1.0 / x


def lt(a: float, b: float) -> float:
    """Check if a is less than b"""
    return 1.0 if a < b else 0.0


def max(a: float, b: float) -> float:
    """Take the max of a and b"""
    return a if a > b else b


def mul(a: float, b: float) -> float:
    """Get the product of a and b"""
    return a * b


def neg(x: float) -> float:
    """Get the negative value of number"""
    return -x * 1.0


def relu(x: float) -> float:
    """Calculate is relu function"""
    return 0.0 if x <= 0 else x


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    if x > 0:
        return y
    else:
        return 0.0


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return (-1.0 / x**2) * y


def log(x: float) -> float:
    """Computes natural logarithmic"""
    return math.log(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return (1.0 / x) * y


#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

T = TypeVar("T")
R = TypeVar("R")
U = TypeVar("U")


def map(func: Callable[[T], R], iterable: Iterable[T]) -> List[R]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [func(item) for item in iterable]


def zipWith(
    func: Callable[[T, U], R], iter1: Iterable[T], iter2: Iterable[U]
) -> Iterator[R]:
    """Higher-order function that combines elements from two iterables using a given function"""
    iterator1 = iter(iter1)
    iterator2 = iter(iter2)
    while True:
        try:
            yield func(next(iterator1), next(iterator2))
        except StopIteration:
            return


def reduce(func: Callable[[R, T], R], iterable: Iterable[T], inital: R) -> R:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    final_val = inital
    for element in iterable:
        final_val = func(final_val, element)
    return final_val


# Small practice library of elementary higher-order functions.
def negList(l: Iterable) -> Iterable:
    """Negate all elements in a list using map"""
    return map(lambda x: -x, l)


def addLists(l1: Iterable, l2: Iterable) -> Iterable:
    """Add corresponding elements from two lists using zipWith"""
    return list(zipWith(lambda x, y: x + y, l1, l2))


def sum(l: Iterable) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(lambda x, y: x + y, l, 0)


def prod(l: Iterable) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(lambda x, y: x * y, l, 1)


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
