from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import Partial

from ..WoS_utils import distance_polylines_index


@jit
def boundary_polylines() -> Array:
    """Return the polylines defining the unit square."""
    # counterclockwise orientation
    return jnp.array(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    )


def boundary_function(boundaries: Array) -> Callable[[Array], Array]:
    """Turn the boundary values into a function."""
    # [bottom, right, top, left]
    boundary_dirichlet = boundary_polylines()
    # [top, bottom, left, right]
    index = jnp.array([1, 3, 0, 2])

    @jit
    def g(x: Array) -> Array:
        """Boundary value function."""
        _, (_, j) = distance_polylines_index(x, boundary_dirichlet)
        return boundaries[index[j]]

    return Partial(g)
