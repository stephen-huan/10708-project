from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import Partial

from .WoS_utils import distance_polylines_index

EMPTY = jnp.nan  # empty value


@partial(jit, static_argnums=0)
def initialize_solution_square(
    n: int, boundaries: Array | None = None
) -> Array:
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


@partial(jit, static_argnums=(0, 1))
def initialize_solution_rectangle(
    nx: int, ny: int, boundaries: Array | None = None
) -> Array:
    """
    Initialize the solution matrix with the boundary values.

    Args:
        nx: number of grid points in x direction
        ny: number of grid points in y direction
        boundaries: list of boundary values [top, bottom, left, right]
    Returns:
        solution: initialized solution matrix
    """
    if boundaries is None:
        boundaries = jnp.zeros(4)
    mean_value = jnp.mean(boundaries)
    # n + 2 to include the boundary values
    solution = mean_value * jnp.ones((ny + 2, nx + 2))

    # set the boundary values
    solution = solution.at[0, :].set(boundaries[0])  # top
    solution = solution.at[-1, :].set(boundaries[1])  # bottom
    solution = solution.at[:, 0].set(boundaries[2])  # left
    solution = solution.at[:, -1].set(boundaries[3])  # right

    return solution


@partial(jit, static_argnums=(0, 1))
def initialize_solution_ushape(
    nx: int, ny: int, boundaries: Array | None = None
) -> tuple[Array, Array]:
    """
    Initialize the solution matrix with the boundary values.
    Assume the 'neck' of the U is 1/4 the domain height and 1/2 domain width
    Let -1 denote out of domain values

    Args:
        nx: number of grid points in x direction
        ny: number of grid points in y direction
        boundaries: list of boundary values [top, bottom, left, right]
    Returns:
        solution: initialized solution matrix
        mask: boolean mask indicating places where the solution is not defined
    """
    if boundaries is None:
        boundaries = jnp.zeros(4)
    mean_value = jnp.mean(boundaries)
    # n + 2 to include the boundary values
    solution = mean_value * jnp.ones((ny + 2, nx + 2))
    mask = jnp.ones((ny + 2, nx + 2))

    # set the boundary values
    solution = solution.at[0, :].set(boundaries[0])  # top
    mask = mask.at[0, :].set(0)
    solution = solution.at[-1, :].set(boundaries[1])  # bottom
    mask = mask.at[-1, :].set(0)
    solution = solution.at[:, 0].set(boundaries[2])  # left
    mask = mask.at[:, 0].set(0)
    solution = solution.at[:, -1].set(boundaries[3])  # right
    mask = mask.at[:, -1].set(0)

    # set the U shape
    neck_yend = round(3 * ny / 4)
    neck_xstart = round(nx / 4)
    neck_xend = round(3 * nx / 4)

    solution = solution.at[:neck_yend, neck_xstart:neck_xend].set(EMPTY)
    mask = mask.at[:neck_yend, neck_xstart:neck_xend].set(0)
    solution = solution.at[:neck_yend, neck_xstart].set(0)
    mask = mask.at[:neck_yend, neck_xstart].set(0)
    solution = solution.at[:neck_yend, neck_xend].set(0)
    mask = mask.at[:neck_yend, neck_xend].set(0)
    solution = solution.at[neck_yend, neck_xstart : neck_xend + 1].set(0)
    mask = mask.at[neck_yend, neck_xstart : neck_xend + 1].set(0)

    return solution, mask


@partial(jit, static_argnums=(0, 1))
def initialize_solution_circle(
    nx: int, ny: int, boundaries: Array | None = None, eps: int = 0
) -> tuple[Array, Array]:
    """
    Initialize the solution matrix with the boundary values.
    Fix the radius of the circle to be 1/2 the domain width
    Let negative values denote out of domain values

    Args:
        nx: number of grid points in x direction
        ny: number of grid points in y direction
        boundaries: list of boundary values [top, bottom]
    Returns:
        solution: initialized solution matrix
        mask: boolean mask indicating places where the solution is not defined
    """
    if boundaries is None:
        boundaries = jnp.zeros(2)
    mean_value = jnp.mean(boundaries)

    solution = mean_value * jnp.ones((ny, nx))
    mask = jnp.ones((ny, nx))

    center = jnp.array([(nx - 1) / 2, (ny - 1) / 2])
    r2 = jnp.square((nx - 1) / 2)

    def body_fun(i, state: tuple[Array, Array]) -> tuple[Array, Array]:
        """Update the solution and mask."""

        def body_fun(j, state: tuple[Array, Array]) -> tuple[Array, Array]:
            """Update the solution and mask."""
            solution, mask = state
            p = center - jnp.array([i, j])
            d2 = jnp.inner(p, p)

            def true_fun(solution: Array, mask: Array) -> tuple[Array, Array]:
                """Update if out of range."""
                solution = solution.at[i, j].set(EMPTY)
                solution = jnp.where(
                    d2 < r2 + eps,
                    solution.at[i, j].set(
                        jnp.where(i > center[1], boundaries[0], boundaries[1])
                    ),
                    solution,
                )
                return solution, mask.at[i, j].set(0)

            return lax.cond(
                d2 > r2 - eps,
                true_fun,
                lambda solution, mask: (solution, mask),
                solution,
                mask,
            )

        return lax.fori_loop(0, nx, body_fun, state)

    return lax.fori_loop(0, ny, body_fun, (solution, mask))


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


problems = [
    (
        initialize_solution_square(100, boundaries=jnp.array([0, 0, 100, 0])),
        "square",
    ),
    (
        initialize_solution_rectangle(
            100, 10, boundaries=jnp.array([0, 50, 100, 0])
        ),
        "rectangle",
    ),
    (
        initialize_solution_circle(
            100, 100, boundaries=jnp.array([0, 100]), eps=55
        ),
        "circle",
    ),
    (
        initialize_solution_ushape(
            100, 100, boundaries=jnp.array([0, 50, 100, 0])
        ),
        "ushape",
    ),
]
