"""
Testing FD Methods (Gauss-Seidel/Jacobi) on Different Shapes
"""

from pathlib import Path

import jax.numpy as jnp
import seaborn as sns
from jax import Array

from elliptic.problems import (
    initialize_solution_circle,
    initialize_solution_rectangle,
    initialize_solution_square,
    initialize_solution_ushape,
)
from elliptic.solvers import gauss_seidel

figures = Path(__file__).parent / "figures" / "fd"
figures.mkdir(parents=True, exist_ok=True)


def plot_matrix(A: Array, path: Path | str | None = None) -> None:
    """Visualize the matrix."""
    ax = sns.heatmap(
        data=A,
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        square=True,
    )
    if path is not None:
        ax.figure.savefig(path)  # pyright: ignore
    ax.figure.clear()  # pyright: ignore


if __name__ == "__main__":
    iters = 2000

    for problem, name in [
        (
            initialize_solution_square(
                100, boundaries=jnp.array([0, 0, 100, 0])
            ),
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
    ]:
        x, mask = problem if isinstance(problem, tuple) else (problem, None)
        plot_matrix(x, path=figures / f"{name}_init.png")
        solution = gauss_seidel(x, n_iters=iters, mask=mask)
        plot_matrix(solution, path=figures / f"{name}_solution.png")
