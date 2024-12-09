from pathlib import Path

import jax
from jax import random

from elliptic.plotting import plot_mesh
from elliptic.problems import (
    boundary_function,
    boundary_polylines,
    get_matrix,
    problems,
)
from elliptic.solvers import wos_domain, wos_sphere

# enable int64/float64
jax.config.update("jax_enable_x64", True)
rng = random.key(0)

figures = Path(__file__).parent / "figures" / "bem"
figures.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for problem in problems:
        A, L, V, mesh = get_matrix(problem=problem, n=20, res=10)
        points = V.tabulate_dof_coordinates()
        boundary_dirichlet = boundary_polylines(problem)
        g = boundary_function(problem)
        rng, subkey = random.split(rng)
        if problem != "circle":
            values = wos_domain(
                subkey, points, boundary_dirichlet, g, n_walks=100
            )
        else:
            values = wos_sphere(
                subkey, points, boundary_dirichlet, g, n_walks=100
            )
        print(values)
        print(points)

        plot_mesh(
            problem, values, points, path=figures / f"{problem}_solution.png"
        )
