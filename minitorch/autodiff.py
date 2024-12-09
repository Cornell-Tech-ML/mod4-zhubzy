from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals_list = list(vals)

    # Create two copies of the input values
    vals_plus = list(vals_list)
    vals_minus = list(vals_list)

    # Modify the argument of interest by epsilon
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Compute f(x + epsilon) and f(x - epsilon)
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Compute the central difference
    deriv = (f_plus - f_minus) / (2 * epsilon)

    return deriv


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative with respect to the variable.

        Args:
        ----
            x: The derivative value to be accumulated.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns True if the variable is a leaf node in the computation graph, False otherwise."""
        ...

    def is_constant(self) -> bool:
        """Returns True if the variable is a constant node in the computation graph, False otherwise."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of the parent variables in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the chain rule for backpropagation.

        Args:
        ----
            d_output: The derivative of the output with respect to the current variable.

        Returns:
        -------
            An iterable of tuples containing the parent variables and their respective derivatives.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    stack = []

    def dfs(node: Variable) -> None:
        if node.is_constant():
            return
        for parent in node.parents:
            if parent.unique_id not in visited:
                visited.add(parent.unique_id)
                dfs(parent)
        if not node.is_leaf():
            stack.append(node)

    dfs(variable)
    return stack[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_computation_graph = list(topological_sort(variable))
    id_to_var = {v.unique_id: (v, 0) for v in sorted_computation_graph}

    assert sorted_computation_graph[0] == variable, "Topological sort failed"
    id_to_var[sorted_computation_graph[0].unique_id] = (variable, deriv)

    for _, v in enumerate(sorted_computation_graph):
        current_deriv = id_to_var[v.unique_id][1]
        for p, parent_out in v.chain_rule(current_deriv):
            id_to_var.setdefault(p.unique_id, (v, 0))
            if p.is_leaf():
                p.accumulate_derivative(parent_out)
            else:
                id_to_var[p.unique_id] = (p, id_to_var[p.unique_id][1] + parent_out)


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors used during backpropagation."""
        return self.saved_values
