from pathlib import Path
from typing import Any

import dolfin
import matplotlib.pyplot as plt
import seaborn as sns
from jax import Array


def plot_dolfin(A: Any, path: Path | str | None = None) -> None:
    """Wrap dolfin's plot."""
    dolfin.plot(A)
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, pad_inches=0)
    plt.gcf().clear()


def plot_matrix(A: Array, path: Path | str | None = None) -> None:
    """Visualize the matrix."""
    ax = sns.heatmap(
        data=A,
        xticklabels=False,
        yticklabels=False,
        cmap="viridis",
        vmin=0,
        vmax=100,
        square=True,
    )
    plt.tight_layout()
    if path is not None:
        ax.figure.savefig(path, pad_inches=0)  # pyright: ignore
    ax.figure.clear()  # pyright: ignore
