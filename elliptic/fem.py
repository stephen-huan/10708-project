import numpy as np
from dolfin import ds  # pyright: ignore
from dolfin import dx  # pyright: ignore
from dolfin import (
    DOLFIN_EPS,
    Constant,
    DirichletBC,
    FunctionSpace,
    Mesh,
    Point,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    assemble,
    grad,
    inner,
)
from mshr import Circle, Rectangle, generate_mesh
from scipy.sparse import csr_array


def get_matrix(
    problem: str, n: int = 100, res: int = 25, save: bool = False
) -> tuple[csr_array, np.ndarray, FunctionSpace, Mesh]:
    """Get the finite element matrix for the problem geometry."""
    if problem == "square":
        mesh = UnitSquareMesh(n, n)
    elif problem == "circle":
        # center, radius, segments of circle
        circle = Circle(Point(0.0, 0.0), 1, 40)
        # mesh, resolution
        mesh = generate_mesh(circle, res)
    elif problem == "ushape":
        # lower left, upper right
        square = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
        # lower left, upper right
        u_void = Rectangle(Point(0.25, 0.25), Point(0.75, 1.0))
        u_shape = square - u_void
        # mesh, resolution
        mesh = generate_mesh(u_shape, res)
    else:
        raise ValueError(f"Invalid problem geometry {problem}.")

    V = FunctionSpace(mesh, "Lagrange", 1)

    # Square Boundaries:
    def left_boundary(x):
        return x[0] < DOLFIN_EPS

    def right_boundary(x):
        return x[0] > 1.0 - DOLFIN_EPS

    def top_boundary(x):
        return x[1] > 1.0 - DOLFIN_EPS

    def bottom_boundary(x):
        return x[1] < DOLFIN_EPS

    # Circle Boundaries:
    # eps needs to be higher than DOLFIN_EPS since mesh is too coarse
    # for DOLFIN_EPS. May need to decrease this when going to higher res
    eps = 1e-2

    def top_circumference_boundary(x):
        # apply boundary on the entire circumference
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        # condition here is 1 - eps < r < 1 + eps and y > 0, which
        # grabs points that are on circuference and above x-axis
        return r > 1.0 - eps and r < 1.0 + eps and x[1] > 0.0

    def bottom_circumference_boundary(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        # condition here is 1 - eps < r < 1 + eps and y > 0, which
        # grabs points that are on circuference and below x-axis
        return r > 1.0 - eps and r < 1.0 + eps and x[1] < 0.0

    # Define boundary condition
    u0 = Constant(0.0)
    uboundary = Constant(100.0)

    if problem == "square":
        bc_left = DirichletBC(V, uboundary, left_boundary)
        bc_right = DirichletBC(V, u0, right_boundary)
        bc_top = DirichletBC(V, u0, top_boundary)
        bc_bottom = DirichletBC(V, u0, bottom_boundary)
        bcs = [bc_left, bc_right, bc_top, bc_bottom]
    elif problem == "circle":
        bc_top = DirichletBC(V, uboundary, top_circumference_boundary)
        bc_bottom = DirichletBC(V, u0, bottom_circumference_boundary)
        bcs = [bc_top, bc_bottom]
    elif problem == "ushape":
        bc_left = DirichletBC(V, u0, left_boundary)
        bc_right = DirichletBC(V, u0, right_boundary)
        bc_top = DirichletBC(V, uboundary, top_boundary)
        bc_bottom = DirichletBC(V, uboundary, bottom_boundary)
        bcs = [bc_left, bc_right, bc_top, bc_bottom]

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0.0)
    g = Constant(0.0)

    Ak = assemble(inner(grad(u), grad(v)) * dx)
    L = assemble(f * v * dx + g * v * ds)  # type: ignore

    for bc in bcs:
        bc.apply(Ak, L)

    Ak_np = Ak.array()
    L_np = L.get_local()
    # is a list of the coordinates of each of the solution points.
    # The ith entry in coord_map is the coordinate of
    # the ith solution point in x after solving Ax=b
    coord_map = V.tabulate_dof_coordinates()

    if save:
        np.save("data/Ak.npy", Ak_np)
        np.save("data/L.npy", L_np)
        np.save("data/coord_map.npy", coord_map)

    Ak_np = csr_array(Ak_np)
    Ak_np.prune()
    return Ak_np, L_np, V, mesh
