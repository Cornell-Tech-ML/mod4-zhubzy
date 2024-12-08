from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def backward(cls, ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        return cls._backward(ctx, d_output)

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given input values.

        Args:
        ----
            *vals (ScalarLike): Input values.

        Returns:
        -------
            Scalar: Result of applying the scalar function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a,) = ctx.saved_values
        return -d_output / (a * a)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a,) = ctx.saved_values
        s = operators.sigmoid(a)
        return d_output * s * (1 - s)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal to function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass of the multiplication function.

        Args:
        ----
            ctx (Context): The autodiff context.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the multiplication.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass of the scalar function.

        Args:
        ----
            ctx (Context): The autodiff context.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables.

        """
        return 0.0, 0.0
