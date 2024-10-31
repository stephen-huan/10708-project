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

from elliptic import gauss_seidel, jacobi, walk_on_spheres
from elliptic.pde_utils import (
    boundary_function,
    boundary_polylines,
    grid,
    wos_domain,
)
from elliptic.utils import initialize_solution, relative_l2_loss

# enable int64/float64
jax.config.update("jax_enable_x64", True)
rng = random.key(0)

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # use Gauss-Seidel with a large grid size and
    # n_iters to get a good ground truth accuracy
    n = 100
    n_iters = 5000
    boundaries = jnp.array([0, 0, 100, 0])  # [top, bottom, left, right]
    solution = initialize_solution(n, boundaries)

    rng, subkey = random.split(rng)
    m = n + 2
    points = grid(m * m)
    g = boundary_function(boundaries)
    boundary_dirichlet = boundary_polylines()
    wos = partial(
        wos_domain,
        x=points,
        boundary_dirichlet=boundary_dirichlet,
        g=g,
        eps=1e-4,
    )
    wos_solution = wos(subkey, n_walks=100)

    # run Gauss-Seidel
    solution = gauss_seidel(solution, n_iters)

    ground_truth = jnp.copy(solution)

    ax = sns.heatmap(
        data=solution, xticklabels=False, yticklabels=False, cmap="viridis"
    )
    ax.figure.savefig(figures / "pde.png")  # pyright: ignore
    ax.figure.savefig(figures / "pde.pdf")  # pyright: ignore
    ax.figure.clear()  # pyright: ignore

    fig, ax = plt.subplots()
    ax = sns.heatmap(
        data=wos_solution,
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        ax=ax,
    )
    ax.figure.savefig(figures / "wos_pde.png")  # pyright: ignore
    ax.figure.savefig(figures / "wos_pde.pdf")  # pyright: ignore
    ax.figure.clear()  # pyright: ignore

    # experiment with runtimes and accuracies

    data = {"iterations": [], "runtime": [], "accuracy": [], "method": []}
    methods = [
        ("Jacobi", lambda _, x, n_iters: jacobi(x, n_iters)),
        ("Gauss-Seidel", lambda _, x, n_iters: gauss_seidel(x, n_iters)),
        ("WoS", lambda rng, _, n_iters: wos(rng, n_walks=n_iters // 50)),
    ]

    iterations = 100 * 2 ** jnp.arange(6)
    for i, n_iters in enumerate(iterations):
        for method_name, method in methods:
            solution = initialize_solution(n, boundaries)
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
            data["iterations"].append(int(n_iters))
            data["runtime"].append(toc - tic)
            data["accuracy"].append(
                float(relative_l2_loss(solution, ground_truth))
            )
            data["method"].append(method_name)

    data = pd.DataFrame(data)

    ax = sns.lineplot(data=data, x="iterations", y="runtime", hue="method")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Runtime")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Runtime (s)")
    ax.figure.savefig(figures / "runtime.png")  # pyright: ignore
    ax.figure.savefig(figures / "runtime.pdf")  # pyright: ignore
    ax.figure.clear()  # pyright: ignore

    ax = sns.lineplot(data=data, x="iterations", y="accuracy", hue="method")
    ax.set_yscale("log")
    ax.set_title("Relative L2 loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Relative L2 loss")
    plt.savefig(figures / "loss.png")
    plt.savefig(figures / "loss.pdf")
    ax.figure.clear()  # pyright: ignore
