from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.numpy.linalg import norm

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
    """Distance from x to the closest point on the polylines."""
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
def distance_polylines_index(
    x: Array, polylines: Array
) -> tuple[Array, Array]:
    """Distance from x to the closest point on the polylines."""
    return lax.fori_loop(
        0,
        polylines.shape[0],
        lambda i, s: lax.fori_loop(
            0,
            (segments := polylines[i]).shape[0] - 1,
            lambda j, s: lax.cond(
                (d := norm(x - closest_point(x, segments[j], segments[j + 1])))
                <= s[0],
                lambda: (d, (i, j)),
                lambda: s,
            ),
            s,
        ),
        (jnp.inf, (0, 0)),
    )
