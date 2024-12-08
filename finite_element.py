""""
Matrix Methods

First use Fenics to assemble system matrix and
forcing vector, then export to numpy array.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from dolfin import Function

from elliptic.fem import get_matrix
from elliptic.plotting import plot_dolfin
from elliptic.problems import problems

figures = Path(__file__).parent / "figures" / "fem"
figures.mkdir(parents=True, exist_ok=True)

save = False

if __name__ == "__main__":
    for _, name in problems:
        A, L, V, mesh = get_matrix(problem=name, n=20, res=10, save=save)

        # either sparse or regular solve
        # u_np = scipy.linalg.solve(A, L)
        u_np = scipy.sparse.linalg.spsolve(A, L)
        print(max(u_np), min(u_np))

        plt.imshow(A.todense())
        plt.savefig(figures / f"{name}_Ak.png")
        plt.clf()

        plt.imshow(np.expand_dims(L, axis=1), aspect="auto")
        plt.savefig(figures / f"{name}_L.png")
        plt.clf()

        plot_dolfin(mesh, path=figures / f"{name}_mesh.png")  # pyright: ignore

        # defines function on Mesh Space V
        u = Function(V)
        # set the values of the function to the values of u_np
        u.vector().set_local(u_np)

        print(
            f"Number of DoFs: {V.dim()}, "
            + f"mesh vertices: {mesh.num_vertices()}, "  # pyright: ignore
            + f"size of unodalVals: {u_np.shape}"
        )

        plot_dolfin(u, path=figures / f"{name}_solution.png")
