"""
Testing FD Methods (Gauss-Seidel/Jacobi) on Different Shapes
"""

from pathlib import Path

from elliptic.plotting import plot_matrix
from elliptic.problems import problems
from elliptic.solvers import gauss_seidel

figures = Path(__file__).parent / "figures" / "fd"
figures.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    iters = 2000

    for problem, name in problems:
        x, mask = problem if isinstance(problem, tuple) else (problem, None)
        plot_matrix(x, path=figures / f"{name}_init.png")
        solution = gauss_seidel(x, n_iters=iters, mask=mask)
        plot_matrix(solution, path=figures / f"{name}_solution.png")
