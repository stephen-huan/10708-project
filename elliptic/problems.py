from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import Partial

from .WoS_utils import distance_polylines_index


@partial(jit, static_argnums=0)
def initialize_solution(n: int, boundaries: Array | None = None) -> Array:
    """
    Initialize the solution matrix with the boundary values.

    Args:
        n: number of grid points in each direction
        boundaries: list of boundary values [top, bottom, left, right]
    Returns:
        solution: initialized solution matrix
    """
    if boundaries is None:
        boundaries = jnp.zeros(4)
    mean_value = jnp.mean(boundaries)
    # n + 2 to include the boundary values
    solution = mean_value * jnp.ones((n + 2, n + 2))

    # set the boundary values
    solution = solution.at[0, :].set(boundaries[0])  # top
    solution = solution.at[-1, :].set(boundaries[1])  # bottom
    solution = solution.at[:, 0].set(boundaries[2])  # left
    solution = solution.at[:, -1].set(boundaries[3])  # right

    return solution


@jit
def boundary_polylines() -> Array:
    """Return the polylines defining the unit square."""
    # counterclockwise orientation
    return jnp.array(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    )


def boundary_function(boundaries: Array) -> Callable[[Array], Array]:
    """Turn the boundary values into a function."""
    # [bottom, right, top, left]
    boundary_dirichlet = boundary_polylines()
    # [top, bottom, left, right]
    index = jnp.array([1, 3, 0, 2])

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        _, (_, j) = distance_polylines_index(x, boundary_dirichlet)
        return boundaries[index[j]]

    return Partial(g)
