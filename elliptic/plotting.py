from pathlib import Path

import seaborn as sns
from jax import Array


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
