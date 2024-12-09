from pathlib import Path

import jax
from jax import random

from elliptic.plotting import plot_mesh
from elliptic.problems import get_matrix, problems
from elliptic.solvers import wos

# enable int64/float64
jax.config.update("jax_enable_x64", True)
rng = random.key(0)

figures = Path(__file__).parent / "figures" / "bem"
figures.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for problem in problems:
        A, L, V, mesh = get_matrix(problem=problem, n=20, res=10)
        points = V.tabulate_dof_coordinates()
        rng, subkey = random.split(rng)
        values = wos(subkey, problem, points, n_walks=100)

        plot_mesh(
            problem, values, points, path=figures / f"{problem}_solution.png"
        )
