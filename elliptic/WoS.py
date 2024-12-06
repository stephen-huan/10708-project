from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, jit, lax, random

from .WoS_utils import distance_polylines, sphere

# https://www.cs.cmu.edu/~kmcrane/Projects/WalkOnStars/WoSt-tutorial.pdf

KeyArray = Array


@jit
def walk_on_spheres(
    rng: KeyArray,
    x: Array,
    boundary_dirichlet: Array,
    g: Callable[[Array], Array],
    n_walks: int = 1000,
    max_steps: float | Array = jnp.inf,
    eps: float | Array = 1e-6,
) -> Array:
    """The walk on spheres (WoS) algorithm."""
    d = x.shape[0]

    def body_fun(_, state: tuple[KeyArray, Array]) -> tuple[KeyArray, Array]:
        """Inner loop of the WoS algorithm."""
        rng, value = state
        rng, xp, _ = lax.while_loop(
            lambda state: (
                distance_polylines(state[1], boundary_dirichlet) > eps
            )
            & (state[2] < max_steps),
            lambda state: (
                random.split(state[0])[0],
                state[1]
                + distance_polylines(state[1], boundary_dirichlet)
                * sphere(random.split(state[0])[1], d),
                state[2] + 1,
            ),
            (rng, x, 0),
        )
        return rng, value + g(xp)

    return (
        lax.fori_loop(0, n_walks, body_fun, (rng, jnp.zeros(())))[1] / n_walks
    )
