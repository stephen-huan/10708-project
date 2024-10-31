from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    plt.plot(range(1, len(l2_losses) + 1), l2_losses, marker="o")
    plt.gcf().set_size_inches(15, 6)
    plt.xlabel("Number of Random Walks")
    plt.ylabel("L2 Loss")
    plt.title("Convergence of Monte Carlo Matrix Inversion")
    plt.grid(True)
    if show:
        plt.show()
    plt.savefig(figures / "monte_carlo.png")
    plt.savefig(figures / "monte_carlo.pdf")


if __name__ == "__main__":
    # Run the simple example
    simple_example()
