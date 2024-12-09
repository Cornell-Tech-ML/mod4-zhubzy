from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling"""
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # First reshape to split height and width
    # From: batch x channel x height x width
    # To: batch x channel x new_height x kh x new_width x kw
    inner = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute to get kernel dims together
    # From: batch x channel x new_height x kh x new_width x kw
    # To: batch x channel x new_height x new_width x kh x kw
    inner = inner.contiguous().permute(0, 1, 2, 4, 3, 5)

    # Combine kernel dims
    # From: batch x channel x new_height x new_width x kh x kw
    # To: batch x channel x new_height x new_width x (kh * kw)
    return (
        inner.contiguous().view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return tiled.mean(4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -99999)


def argmax(input: Tensor, dim: Tensor) -> Tensor:
    """Compute the argmax as a 1-hot tensor."""
    return max_reduce(input, int(dim.item())) == input


class Max(Function):
    """Max operator implementation"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max operation"""
        ctx.save_for_backward(argmax(input, dim))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max operation"""
        (argmax,) = ctx.saved_values
        return grad_output * argmax, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction on tensor."""

    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax as a tensor."""
    # Subtract max for numerical stability
    x = input - max(input, dim)
    exp_x = x.exp()
    return exp_x / exp_x.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log of softmax as a tensor using the LogSumExp trick."""
    # Get the max for the LogSumExp trick
    max_val = max(input, dim)

    # Compute exp(x - max(x)) for stability
    stable_exp = (input - max_val).exp()

    # Compute log(sum(exp(x - max(x))))
    log_sum_exp_shifted = stable_exp.sum(dim).log()

    # Add max back: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    log_sum_exp = max_val + log_sum_exp_shifted

    # Final result: x - LogSumExp(x)
    return input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D.

    Args:
        input: input tensor of shape (batch, channel, height, width)
        kernel: tuple of (kernel_height, kernel_width)

    Returns:
        Max pooled tensor
    """
    batch, channel, _, _ = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise."""
    if ignore:
        return input

    # Generate mask of 0s and 1s with keep probability (1-prob)
    rand_values = rand(input.shape, input.backend, requires_grad=False)

    # Create binary mask (1 for keep, 0 for drop) based on keep probability
    mask = rand_values > prob

    return input * mask
