from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap
from jax.numpy.linalg import norm
from jax.tree_util import Partial

from .WoS import closest_point, walk_on_spheres

KeyArray = Array


@partial(jit, static_argnums=(0, 3))
def grid(n: int, a: float = 0, b: float = 1, d: int = 2) -> Array:
    """Generate n points evenly spaced in a [a, b]^d hypercube."""
    spaced = jnp.linspace(a, b, round(n ** (1 / d)))
    cube = (spaced,) * d
    return jnp.stack(jnp.meshgrid(*cube), axis=-1).reshape(-1, d)


@jit
def boundary_polylines() -> Array:
    """Return the polylines defining the unit square."""
    # counterclockwise orientation
    return jnp.array(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    )


@jit
def distance_polylines(x: Array, polylines: Array) -> tuple[Array, Array]:
    """Distance from x to the cloest point on the polylines."""
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


def boundary_function(boundaries: Array) -> Callable[[Array], Array]:
    """Turn the boundary values into a function."""
    # [bottom, right, top, left]
    boundary_dirichlet = boundary_polylines()
    # [top, bottom, left, right]
    index = jnp.array([1, 3, 0, 2])

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        _, (_, j) = distance_polylines(x, boundary_dirichlet)
        return boundaries[index[j]]

    return Partial(g)


@jit
def wos_domain(
    rng: KeyArray,
    x: Array,
    boundary_dirichlet: Array,
    g: Callable[[Array], Array],
    n_walks: int = 1000,
    max_steps: float | Array = jnp.inf,
    eps: float | Array = 1e-6,
):
    """Run walk on spheres on an entire domain."""
    wos = partial(
        walk_on_spheres,
        boundary_dirichlet=boundary_dirichlet,
        g=g,
        n_walks=n_walks,
        max_steps=max_steps,
        eps=eps,
    )
    n, d = x.shape
    m = round(n ** (1 / d))
    subkeys = random.split(rng, num=x.shape[0])
    return vmap(wos)(subkeys, x).reshape((m,) * d)
