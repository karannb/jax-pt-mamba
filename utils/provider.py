import jax.numpy as jnp
from jax import vmap, random

from jax._src.basearray import Array
from jax._src.prng import PRNGKeyArray


def shift_point_cloud(
    data: Array, key: PRNGKeyArray, shift_range: jnp.float32 = 0.1
) -> Array:
    """Randomly shift point cloud.
    Input:
      Nx3 array, original point cloud
    Return:
      Nx3 array, shifted point cloud
    """
    shifts = random.uniform(key, (3,), minval=-shift_range, maxval=shift_range)
    data = data + shifts
    return data


batched_shift_point_cloud = vmap(shift_point_cloud, in_axes=(0, 0, None))


def random_scale_point_cloud(
    data: Array,
    key: PRNGKeyArray,
    scale_low: jnp.float32 = 0.8,
    scale_high: jnp.float32 = 1.25,
) -> Array:
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        Nx3 array, original point cloud
    Return:
        Nx3 array, scaled point cloud
    """
    scales = random.uniform(key, (1,), minval=scale_low, maxval=scale_high)
    data = data * scales[:, None, None]
    return data


batched_random_scale_point_cloud = vmap(
    random_scale_point_cloud, in_axes=(0, 0, None, None)
)
