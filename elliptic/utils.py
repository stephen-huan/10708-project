from functools import partial

import jax.numpy as jnp
from jax import Array, jit, vmap


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


@partial(jit, static_argnums=(0, 3))
def grid(n: int, a: float = 0, b: float = 1, d: int = 2) -> Array:
    """Generate n points evenly spaced in a [a, b]^d hypercube."""
    spaced = jnp.linspace(a, b, round(n ** (1 / d)))
    cube = (spaced,) * d
    return jnp.stack(jnp.meshgrid(*cube), axis=-1).reshape(-1, d)


@jit
def _get_closest(x: Array, points: Array, values: Array) -> Array:
    """Get the value corresponding to the closest point to x."""
    return values[jnp.argmin(jnp.sum(jnp.square(points - x), axis=1))]


@jit
def closest_values(x: Array, points: Array, values: Array) -> Array:
    """Get the value corresponding to the closest point to x."""
    return vmap(partial(_get_closest, points=points, values=values))(x)
