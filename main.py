"""
Solving the Laplace equation using finite differences, implementing
the Jacobi and Gauss-Seidel methods. Better ones definitely exist!

https://surface.syr.edu/cgi/viewcontent.cgi?article=1160&context=eecs_techreports
"""

import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import random

from elliptic.plotting import plot_matrix, plot_mesh
from elliptic.problems import get_matrix, initialize_solution, problems
from elliptic.solvers import (
    conjugate_gradient,
    conjugate_residual,
    gauss_seidel,
    jacobi,
    wos,
)
from elliptic.utils import closest_values, grid, relative_l2_loss

# enable int64/float64
jax.config.update("jax_enable_x64", True)
rng = random.key(0)

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)

plot_solutions = False

if __name__ == "__main__":
    for problem in problems:
        n = 100
        res = 75
        n_iters = 5000

        A, L, V, _ = get_matrix(problem, n=n, res=res)
        points = V.tabulate_dof_coordinates()
        # ground truth solution
        ground_truth, _ = conjugate_residual(A, L, maxiter=n_iters)
        print("init", _[-1])
        if plot_solutions:
            plot_mesh(
                problem,
                ground_truth,
                points,
                path=figures / f"{problem}_pde.png",
            )

        solution = initialize_solution(problem, n=n)
        solution, mask = (
            solution if isinstance(solution, tuple) else (solution, None)
        )
        m = n + 2 if problem != "circle" else n
        fd_points = grid(m * m)
        if mask is not None:
            fd_points = fd_points[mask.flatten()]
        # run Gauss-Seidel
        if plot_solutions:
            solution = gauss_seidel(solution, n_iters, mask=mask)
            plot_matrix(solution, path=figures / f"{problem}_gs.png")

        rng, subkey = random.split(rng)
        WoS = partial(wos, problem=problem, x=points, eps=1e-4)
        if plot_solutions:
            wos_solution = WoS(subkey, n_walks=100)
            plot_mesh(
                problem,
                wos_solution,
                points,
                path=figures / f"{problem}_wos.png",
            )

        # experiment with runtimes and accuracies

        data = {"iterations": [], "runtime": [], "accuracy": [], "method": []}
        methods = [
            ("Jacobi", lambda _, x, n_iters: jacobi(x, n_iters, mask=mask)),
            (
                "Gauss-Seidel",
                lambda _, x, n_iters: gauss_seidel(x, n_iters, mask=mask),
            ),
            ("WoS", lambda rng, _, n_iters: WoS(rng, n_walks=n_iters // 50)),
            (
                "CG",
                lambda _, __, n_iters: conjugate_gradient(
                    A, L, maxiter=n_iters
                )[0],
            ),
            (
                "CR",
                lambda _, __, n_iters: conjugate_residual(
                    A, L, maxiter=n_iters
                )[0],
            ),
        ]

        iterations = 100 * 2 ** jnp.arange(6)
        for i, n_iters in enumerate(iterations):
            for method_name, method in methods:
                solution = initialize_solution(problem, n=n)
                solution, mask = (
                    solution
                    if isinstance(solution, tuple)
                    else (solution, None)
                )
                rng, subkey = random.split(rng)
                # remove jit compilation
                if i == 0:
                    jnp.array(
                        method(subkey, jnp.copy(solution), n_iters)
                    ).block_until_ready()
                tic = time.time()
                solution = jnp.array(
                    method(subkey, solution, n_iters)
                ).block_until_ready()
                toc = time.time()
                print(method_name, n_iters, toc - tic)
                if method_name in ["Jacobi", "Gauss-Seidel"]:
                    solution = solution.flatten()
                    if mask is not None:
                        solution = solution[mask.flatten()]
                    loss = relative_l2_loss(
                        solution,
                        closest_values(fd_points, points, ground_truth),
                    )
                else:
                    loss = relative_l2_loss(solution, ground_truth)
                data["iterations"].append(int(n_iters))
                data["runtime"].append(toc - tic)
                data["accuracy"].append(float(loss))
                data["method"].append(method_name)

        data = pd.DataFrame(data)

        ax = sns.lineplot(data=data, x="iterations", y="runtime", hue="method")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"Runtime ({problem})")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Runtime (s)")
        ax.figure.savefig(  # pyright: ignore
            figures / f"{problem}_runtime.png"
        )
        # ax.figure.savefig(  # pyright: ignore
        #     figures / f"{problem}_runtime.pdf"
        # )
        ax.figure.clear()  # pyright: ignore

        ax = sns.lineplot(
            data=data, x="iterations", y="accuracy", hue="method"
        )
        ax.set_yscale("log")
        ax.set_title(f"Relative L2 loss ({problem})")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Relative L2 loss")
        plt.savefig(figures / f"{problem}_loss.png")
        # plt.savefig(figures / f"{problem}_loss.pdf")
        ax.figure.clear()  # pyright: ignore
