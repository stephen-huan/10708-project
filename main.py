"""
Solving the Laplace equation using finite differences, implementing
the Jacobi and Gauss-Seidel methods. Better ones definitely exist!

https://surface.syr.edu/cgi/viewcontent.cgi?article=1160&context=eecs_techreports
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from jax import random

from elliptic import gauss_seidel, jacobi
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

    # run Gauss-Seidel
    solution = gauss_seidel(solution, n_iters)

    ground_truth = jnp.copy(solution)

    ax = sns.heatmap(
        data=solution, xticklabels=False, yticklabels=False, cmap="viridis"
    )
    ax.figure.savefig(figures / "pde.png")  # pyright: ignore
    ax.figure.savefig(figures / "pde.pdf")  # pyright: ignore
    ax.figure.clear()  # pyright: ignore

    # experiment with runtimes and accuracies

    data = {"iterations": [], "runtime": [], "accuracy": [], "method": []}
    methods = [
        ("Jacobi", jacobi),
        ("Gauss-Seidel", gauss_seidel),
    ]

    iterations = 100 * 2 ** jnp.arange(6)
    for n_iters in iterations:
        for method_name, method in methods:
            solution = initialize_solution(n, boundaries)
            # remove jit compilation
            method(jnp.copy(solution), n_iters).block_until_ready()
            tic = time.time()
            solution = method(solution, n_iters).block_until_ready()
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
