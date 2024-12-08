from .boundary_element import boundary_function, boundary_polylines
from .finite_difference import initialize_solution
from .finite_element import get_matrix

problems = [
    "square",
    # "rectangle",
    "circle",
    "ushape",
]

__all__ = [
    "get_matrix",
    "problems",
    "initialize_solution",
    "boundary_function",
    "boundary_polylines",
]
