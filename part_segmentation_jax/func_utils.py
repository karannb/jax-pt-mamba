import jax
import flax
from jax import random
import jax.numpy as jnp
from jax._src.basearray import Array

import numpy as np
from typing import List
from flax import linen as nn


def knn(ref: Array, query: Array, k: int):
    """
    Compute k-neighbourhood for each point in query from
    ref.

    Args:
        ref: (B, N, 3)
        This is the set of non-fps sampled points, basically
        the whole point cloud.

        query: (B, fps_num, 3)
        This is the set of fps samples points.

        k: int
        Number of ref points per query points, i.e.,
        what should be considered the neightbourhood of each
        query point.

    Returns:
        idx: (B, fps_num, k)
        Returns the indices of k-closest neighbours of each
        query point.
    """

    dist_matrix = jnp.linalg.norm(query[:, :, None, :] - ref[:, None, :, :], axis=-1)
    inds = jnp.argsort(dist_matrix, axis=-1)
    return inds[:, :, :k]


def sort_select_and_concat(
    operand: Array, indices: List[List], ind_axis: int = -1, axis: int = 1
) -> Array:

    ovr_features = []
    for ind in indices:

        feature = jnp.take_along_axis(
            operand,
            jnp.repeat(ind, repeats=operand.shape[-1], axis=ind_axis),
            axis=axis,
        )
        ovr_features.append(feature)

    return jnp.concatenate(ovr_features, axis=axis)


def custom_transpose(x: Array):
    """
    Courtesy of ChatGPT.
    """

    # Get the total number of dimensions in the array
    num_axes = x.ndim

    # Create a list of axes in order
    axes_order = list(range(num_axes))

    # Swap the last two axes
    axes_order[-1], axes_order[-2] = axes_order[-2], axes_order[-1]

    # Transpose the array according to the new order
    reversed_axes_array = jnp.transpose(x, axes=axes_order)

    return reversed_axes_array


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict) or isinstance(value, flax.core.FrozenDict):
            print_params(value, prefix=prefix + key + "/")
        else:
            print(f"{prefix}{key}: {value.shape}")


def drop_path(
    x: Array, drop_prob: float = 0.0, key: random.PRNGKey = None, training: bool = False
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + random.uniform(key, shape, dtype=x.dtype)
    random_tensor = jnp.floor(random_tensor)  # binarize
    output = jnp.divide(x, keep_prob) * random_tensor
    return output


class DropPathV2(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    drop_prob: float = None

    @nn.compact
    def __call__(self, x, drop_key, training):
        return drop_path(x, self.drop_prob, drop_key, training)


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = jax.random.uniform(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """

    @nn.compact
    def __call__(self, input: Array, *args, **kwargs) -> Array:
        return input


class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param(
            "weight", nn.initializers.ones, (self.d_model,)
        )  # TODO, maybe use setup will be more clear
        normed = x * jax.lax.rsqrt(
            np.mean(np.square(x), axis=-1, keepdims=True) + self.eps
        )
        output = normed * weight
        return output
