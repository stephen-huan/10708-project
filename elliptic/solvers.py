from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap

from .WoS import walk_on_spheres

KeyArray = Array


@partial(jit, donate_argnums=0)
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


@partial(jit, donate_argnums=0)
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
