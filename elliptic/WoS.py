from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.numpy.linalg import norm

# https://www.cs.cmu.edu/~kmcrane/Projects/WalkOnStars/WoSt-tutorial.pdf

KeyArray = Array


@partial(jit, static_argnums=1)
def sphere(rng: KeyArray, d: int) -> Array:
    """Sample a point uniformly from the unit L^2 sphere."""
    z = random.normal(rng, (d,))
    return z / norm(z)


@jit
def closest_point(x: Array, a: Array, b: Array) -> Array:
    """Return the closest point to x on the segment connecting a and b."""
    u = b - a
    t = jnp.clip(jnp.dot(x - a, u) / jnp.dot(u, u), 0, 1)
    return (1 - t) * a + t * b


@jit
def distance_polylines(x: Array, polylines: Array) -> Array:
    """Distance from x to the cloest point on the polylines."""
    return lax.fori_loop(
        0,
        polylines.shape[0],
        lambda i, d: lax.fori_loop(
            0,
            (segments := polylines[i]).shape[0] - 1,
            lambda j, d: jnp.minimum(
                d, norm(x - closest_point(x, segments[j], segments[j + 1]))
            ),
            d,
        ),
        jnp.inf,
    )


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
