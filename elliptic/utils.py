from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax


@jit
def relative_l2_loss(solution: Array, solution_ref: Array) -> Array:
    """
    Compute the relative L2 loss between the solution and the reference.

    Args:
        solution: computed solution
        solution_ref: reference solution
    Returns:
        relative_l2_loss: relative L2 loss
    """
    numerator = jnp.linalg.norm(solution - solution_ref, ord=2)
    denominator = jnp.linalg.norm(solution_ref, ord=2)
    relative_l2_loss = numerator / denominator
    return relative_l2_loss


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
def jacobi(x: Array, n_iters: int = 1) -> Array:
    """Jacobi method for solving an elliptic PDE."""
    n, m = x.shape

    def body_fun(_: int, x: Array) -> Array:
        """Inner loop of the Jacobi method."""
        return lax.fori_loop(
            1,
            n - 1,
            lambda i, xp: lax.fori_loop(
                1,
                m - 1,
                lambda j, xp: xp.at[i, j].set(
                    (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1]) / 4
                ),
                xp,
            ),
            x,
        )

    return lax.fori_loop(0, n_iters, body_fun, x)


@jit
def gauss_seidel(x: Array, n_iters: int = 1) -> Array:
    """Gauss-Seidel method for solving an elliptic PDE."""
    n, m = x.shape

    def body_fun(_: int, x: Array) -> Array:
        """Inner loop of the Gauss-Seidel method."""
        return lax.fori_loop(
            1,
            n - 1,
            lambda i, x: lax.fori_loop(
                1,
                m - 1,
                lambda j, x: x.at[i, j].set(
                    (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1]) / 4
                ),
                x,
            ),
            x,
        )

    return lax.fori_loop(0, n_iters, body_fun, x)
