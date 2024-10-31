"""
Solving the Laplace equation using finite differences, implementing
the Jacobi and Gauss-Seidel methods. Better ones definitely exist!

https://surface.syr.edu/cgi/viewcontent.cgi?article=1160&context=eecs_techreports
"""

import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from elliptic import gauss_seidel, jacobi
from elliptic.utils import initialize_solution, relative_l2_loss

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Use Gauss-Seidel with a large grid size and
    # n_iters to get a good ground truth accuracy
    n = 100
    n_iters = 5000
    boundaries = [0, 0, 100, 0]  # [top, bottom, left, right]
    solution = initialize_solution(n, boundaries)

    # run Gauss-Seidel
    tic = time.time()
    solution = gauss_seidel(solution, n_iters)
    toc = time.time()

    print(f"Gauss Seidel took {toc - tic} seconds to run {n_iters} iterations")
    ground_truth = np.copy(solution)

    plt.savefig(figures / "pde.png")
    plt.savefig(figures / "pde.pdf")

    # experiment with runtimes and accuracies

    # Jacobi
    iterations = [100, 200, 400, 800, 1600, 3200]
    jacobi_runtimes = []
    jacobi_accuracies = []

    for n_iters in iterations:
        solution = initialize_solution(n, boundaries)
        tic = time.time()
        solution = jacobi(solution, n_iters)
        toc = time.time()
        jacobi_runtimes.append(toc - tic)
        jacobi_accuracies.append(relative_l2_loss(solution, ground_truth))
        print(
            f"Jacobi took {toc - tic} seconds to run {n_iters} "
            + f"iterations with relative L2 loss {jacobi_accuracies[-1]}"
        )

    # Gauss-Seidel
    gauss_seidel_runtimes = []
    gauss_seidel_accuracies = []

    for n_iters in iterations:
        solution = initialize_solution(n, boundaries)
        tic = time.time()
        solution = gauss_seidel(solution, n_iters)
        toc = time.time()
        gauss_seidel_runtimes.append(toc - tic)
        gauss_seidel_accuracies.append(
            relative_l2_loss(solution, ground_truth)
        )
        print(
            f"Gauss-Seidel took {toc - tic} seconds to run {n_iters} "
            + f"iterations with relative L2 loss {gauss_seidel_accuracies[-1]}"
        )

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(iterations, jacobi_runtimes, label="Jacobi")
    axs[0].plot(iterations, gauss_seidel_runtimes, label="Gauss Seidel")
    axs[0].set_title("Runtimes")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Runtime (s)")
    axs[0].legend()

    axs[1].plot(iterations, jacobi_accuracies, label="Jacobi")
    axs[1].plot(iterations, gauss_seidel_accuracies, label="Gauss Seidel")
    axs[1].set_title("Relative L2 Loss")
    axs[1].set_xlabel("Iterations")
    axs[1].set_ylabel("Relative L2 Loss")
    axs[1].legend()
    plt.savefig(figures / "runtime_loss.png")
    plt.savefig(figures / "runtime_loss.pdf")
