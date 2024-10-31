from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from elliptic import monte_carlo_matrix_inversion

figures = Path(__file__).parent / "figures"
figures.mkdir(parents=True, exist_ok=True)


def simple_example(show: bool = False) -> None:
    """Very simple test matrix and convergence plot."""
    # Very simple test matrix B
    B = np.array([[2, -1], [-1, 2]])

    # Run the Monte Carlo matrix inversion and track convergence
    _, l2_losses = monte_carlo_matrix_inversion(B, max_steps=100)

    # Plot L2 loss for convergence across the number of random walks
    ax = sns.lineplot(x=range(1, len(l2_losses) + 1), y=l2_losses, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Random Walks")
    ax.set_ylabel("L2 Loss")
    ax.set_title("Convergence of Monte Carlo Matrix Inversion")
    if show:
        plt.show()
    ax.figure.savefig(figures / "monte_carlo.png")  # pyright: ignore
    ax.figure.savefig(figures / "monte_carlo.pdf")  # pyright: ignore


if __name__ == "__main__":
    # Run the simple example
    simple_example()
