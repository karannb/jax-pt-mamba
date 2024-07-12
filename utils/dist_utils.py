import jax
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def reshape_batch_per_device(x, num_devices):
    return jax.tree_util.tree_map(
        partial(reshape_array_per_device, num_devices=num_devices), x
    )


def reshape_array_per_device(x, num_devices):
    batch_size_per_device, ragged = divmod(x.shape[0], num_devices)
    if ragged:
        msg = "batch size must be divisible by device count, got {} and {}."
        raise ValueError(msg.format(x.shape[0], num_devices))
    return x.reshape(
        (
            num_devices,
            batch_size_per_device,
        )
        + (x.shape[1:])
    )
