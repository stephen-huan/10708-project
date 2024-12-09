from pathlib import Path
from typing import Any

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import Array
from matplotlib import tri as mtri
from shapely.geometry import Polygon as sPolygon


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


def _plot_mesh(u: np.ndarray, coords: np.ndarray) -> None:
    """Plot a coordinate mesh."""
    x = coords[:, 0]
    y = coords[:, 1]

    triang = mtri.Triangulation(x, y)
    refiner = mtri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(u, subdiv=1)

    plt.figure()
    plt.gca().set_aspect("equal")
    plt.tricontourf(tri_refi, z_test_refi, levels=50)


def _plot_mesh_ushape(u: np.ndarray, coords: np.ndarray) -> None:
    """Plot a coordinate mesh based on u-shape."""
    # define outline based on coords and mask out triangles outside
    coord_x = coords[:, 0]
    coord_y = coords[:, 1]
    x = np.array(
        [0.0, 0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0]
    )  # outline of Ushape
    y = np.array([0.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 0.0])
    outline = sPolygon(zip(x, y))

    triang = mtri.Triangulation(coord_x, coord_y)
    mask = [
        not outline.contains(sPolygon(zip(coord_x[tri], coord_y[tri])))
        for tri in triang.get_masked_triangles()
    ]
    triang.set_mask(mask)

    refiner = mtri.UniformTriRefiner(triang)
    tri_refi, z_test_refi = refiner.refine_field(u, subdiv=1)

    plt.figure()
    plt.gca().set_aspect("equal")
    plt.tricontourf(tri_refi, z_test_refi, levels=50)


def plot_mesh(
    problem: str,
    u: Array | np.ndarray,
    coords: Array | np.ndarray,
    path: Path | str | None = None,
) -> None:
    """Plot a coordinate mesh."""
    u = np.array(u)
    coords = np.array(coords)

    if problem != "ushape":
        _plot_mesh(u, coords)
    else:
        _plot_mesh_ushape(u, coords)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, pad_inches=0)
    plt.gcf().clear()
