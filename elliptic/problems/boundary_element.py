from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import Partial

from ..WoS_utils import distance_polylines_index


@jit
def boundary_polylines_square() -> Array:
    """Return the polylines defining the unit square."""
    # counterclockwise orientation
    return jnp.array(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    )


def boundary_function_square(boundaries: Array) -> Callable[[Array], Array]:
    """Turn the boundary values into a function."""
    # [bottom, right, top, left]
    boundary_dirichlet = boundary_polylines_square()
    # [top, bottom, left, right]
    index = jnp.array([1, 3, 0, 2])

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        _, (_, j) = distance_polylines_index(x, boundary_dirichlet)
        return boundaries[index[j]]

    return Partial(g)


@jit
def boundary_polylines_circle() -> Array:
    """Return the polylines defining the unit circle."""
    # just the radius
    return jnp.ones(())


def boundary_function_circle(boundaries: Array):
    """Turn the boundary values into a function."""

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        return jnp.where(x[1] > 0, boundaries[0], boundaries[1])

    return Partial(g)


@jit
def boundary_polylines_ushape() -> Array:
    """Return the polylines defining the u-shape."""
    # counterclockwise orientation
    return jnp.array(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.75, 1.0],
                [0.75, 0.25],
                [0.25, 0.25],
                [0.25, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ]
    )


def boundary_function_ushape(boundaries: Array) -> Callable[[Array], Array]:
    """Turn the boundary values into a function."""
    boundary_dirichlet = boundary_polylines_ushape()

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        _, (_, j) = distance_polylines_index(x, boundary_dirichlet)
        return boundaries[j]

    return Partial(g)


def boundary_polylines(problem: str) -> Array:
    """Return the polylines defining the problem."""
    if problem == "square":
        return boundary_polylines_square()
    elif problem == "circle":
        return boundary_polylines_circle()
    elif problem == "ushape":
        return boundary_polylines_ushape()
    else:
        raise ValueError(f"Invalid problem geometry {problem}.")


def boundary_function(problem: str) -> Callable[[Array], Array]:
    """Return the boundary function."""
    if problem == "square":
        return boundary_function_square(jnp.array([0, 0, 100, 0]))
    elif problem == "circle":
        return boundary_function_circle(jnp.array([100, 0]))
    elif problem == "ushape":
        return boundary_function_ushape(jnp.array([50, 0, 0, 0, 0, 0, 0, 100]))
    else:
        raise ValueError(f"Invalid problem geometry {problem}.")
