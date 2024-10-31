from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, jit, random, vmap

from elliptic.utils import boundary_polylines
from elliptic.WoS import distance_polylines

# enable int64/float64
jax.config.update("jax_enable_x64", True)
rng = random.key(0)


@jit
def distance_unitsquare(x: Array) -> Array:
    """Distance to the closest point on the boundary of the unit square."""
    return jnp.min(
        jnp.array(
            [
                jnp.abs(x[0]),
                jnp.abs(1 - x[0]),
                jnp.abs(x[1]),
                jnp.abs(1 - x[1]),
            ]
        )
    )


if __name__ == "__main__":
    boundary = boundary_polylines()

    rng, subkey = random.split(rng)
    n = int(10**6)
    d = 2
    points = random.uniform(rng, (n, d))

    ans1 = vmap(distance_unitsquare)(points)
    ans2 = vmap(partial(distance_polylines, polylines=boundary))(points)
    assert jnp.allclose(ans1, ans2), "distance_polylines wrong on unit square."
