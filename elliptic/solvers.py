from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, random, vmap
from jax.tree_util import Partial
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, cg

from .problems import boundary_function, boundary_polylines
from .WoS import sphere_walk_on_spheres, walk_on_spheres

KeyArray = Array


def _update(x: Array, i: Array, j: Array) -> Array:
    """Core update of finite difference methods."""
    return (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1]) / 4


@partial(jit, donate_argnums=0)
def jacobi(x: Array, n_iters: int = 1, mask: Array | None = None) -> Array:
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
                    _update(x, i, j)
                    if mask is None
                    else jnp.where(mask[i, j], _update(x, i, j), xp[i, j])
                ),
                xp,
            ),
            x,
        )

    return lax.fori_loop(0, n_iters, body_fun, x)


@partial(jit, donate_argnums=0)
def gauss_seidel(
    x: Array, n_iters: int = 1, mask: Array | None = None
) -> Array:
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
                    _update(x, i, j)
                    if mask is None
                    else jnp.where(mask[i, j], _update(x, i, j), x[i, j])
                ),
                x,
            ),
            x,
        )

    return lax.fori_loop(0, n_iters, body_fun, x)


def _conjugate_residual(
    A,
    b,
    x0,
    *,
    rtol: float = 1e-5,
    maxiter: int | float | None = None,
    M: np.ndarray | LinearOperator | None = None,
) -> tuple[np.ndarray, list]:
    """Conjugate residual with residual tracking"""
    if maxiter is None:
        maxiter = float("inf")
    if M is None:
        M = LinearOperator(A.shape, matvec=lambda x: x)  # pyright: ignore

    x = x0
    b = M @ b
    b_norm = np.linalg.norm(b)
    r = b - M @ (A @ x0)
    p = r
    Ar = A @ r
    Ap = Ar
    # track initial residual norm
    residuals = [np.linalg.norm(r)]
    iters = 0

    while iters < maxiter:
        MAp = M @ Ap
        alpha = np.inner(r, Ar) / np.inner(Ap, MAp)
        x_new = x + alpha * p
        r_new = r - alpha * MAp
        # track residual norm
        residuals.append(np.linalg.norm(r_new))
        # check for convergence
        if residuals[-1] <= rtol * b_norm:
            return x_new, residuals
        Ar_new = A @ r_new
        beta = np.inner(r_new, Ar_new) / np.inner(r, Ar)
        p_new = r_new + beta * p
        Ap_new = Ar_new + beta * Ap

        x, r, p, Ap, Ar = x_new, r_new, p_new, Ap_new, Ar_new
        iters += 1

    return x, residuals


def conjugate_residual(
    A,
    b,
    *,
    maxiter: int,
    hermitian: bool = False,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Conjugate residual method."""
    if x0 is None:
        x0 = np.zeros_like(b)
    if hermitian:
        B_diag = np.reciprocal(np.diagonal(A))
        B = A
    else:
        B_diag = np.reciprocal(np.sum(A * A, axis=0))
        B = LinearOperator(
            A.shape, matvec=lambda x: A.T @ (A @ x)  # pyright: ignore
        )
    M = LinearOperator(B.shape, matvec=lambda x: x * B_diag)  # pyright: ignore
    return _conjugate_residual(B, b, x0, rtol=0, maxiter=maxiter, M=M)


def conjugate_gradient(
    A,
    b,
    *,
    maxiter: int,
    hermitian: bool = False,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Conjugate gradient method."""
    if x0 is None:
        x0 = np.zeros_like(b)
    if hermitian:
        B_diag = np.reciprocal(np.diagonal(A))
        B = A
    else:
        B_diag = np.reciprocal(np.sum(A * A, axis=0))
        B = LinearOperator(
            A.shape, matvec=lambda x: A.T @ (A @ x)  # pyright: ignore
        )
    M = LinearOperator(B.shape, matvec=lambda x: x * B_diag)  # pyright: ignore

    residuals = []

    def callback(xk: np.ndarray) -> None:
        """User-provided callback."""
        residuals.append(np.linalg.norm(M @ b - M @ (B @ xk)))

    return (
        cg(B, b, x0, rtol=0, maxiter=maxiter, M=M, callback=callback)[0],
        residuals,
    )


@jit
def wos_domain(
    rng: KeyArray,
    x: Array,
    boundary_dirichlet: Array,
    g: Callable[[Array], Array],
    n_walks: int = 1000,
    max_steps: float | Array = jnp.inf,
    eps: float | Array = 1e-6,
    walk_on_spheres: Callable = walk_on_spheres,
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
    subkeys = random.split(rng, num=x.shape[0])
    return vmap(wos)(subkeys, x)


def wos(
    rng: KeyArray,
    problem: str,
    x: Array,
    n_walks: int = 1000,
    max_steps: float | Array = jnp.inf,
    eps: float | Array = 1e-6,
):
    """Run walk on spheres on an entire domain."""
    return wos_domain(
        rng,
        x,
        boundary_polylines(problem),
        boundary_function(problem),
        n_walks=n_walks,
        max_steps=max_steps,
        eps=eps,
        walk_on_spheres=Partial(
            walk_on_spheres if problem != "circle" else sphere_walk_on_spheres
        ),
    )


def hybrid_wos(
    rng: KeyArray,
    problem: str,
    A: csr_array,
    b: np.ndarray,
    x: Array,
    n_iters: int = 1000,
    max_steps: float | Array = jnp.inf,
    eps: float | Array = 1e-6,
    wos_variance: float = 1,
):
    """Run walk on spheres on an entire domain."""
    # roughly balance runtime
    maxiter = max(int(n_iters * 7 / 8), 1)
    n_walks = max((n_iters - maxiter) // 50, 1)
    wos_solution = wos_domain(
        rng,
        x,
        boundary_polylines(problem),
        boundary_function(problem),
        n_walks=n_walks,
        max_steps=max_steps,
        eps=eps,
        walk_on_spheres=Partial(
            walk_on_spheres if problem != "circle" else sphere_walk_on_spheres
        ),
    )
    var = n_walks / wos_variance
    cr_solution, _ = conjugate_gradient(A, b, maxiter=maxiter, x0=wos_solution)
    mse = np.reciprocal(np.sum(jnp.square(b - A @ cr_solution)))
    return (mse * cr_solution + var * wos_solution) / (mse + var)
