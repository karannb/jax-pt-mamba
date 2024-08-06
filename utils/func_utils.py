# jax imports
import jax
from jax import random
import jax.numpy as jnp
from jax._src import prng
from jax._src.basearray import Array

# other imports
import flax
import numpy as np
from flax import linen as nn
from flax.core import FrozenDict

# from scipy.spatial import cKDTree
from utils.dropout import Dropout
from typing import List, Union, Dict, Any

# type definitions
KeyArray = prng.PRNGKeyArray


def knn(ref: Array, query: Array, k: int) -> Array:
    """
    Compute k-neighbourhood for each point in query from
    ref.

    Args
    ----
        ref: (N, 3)
        This is the set of non-fps sampled points, basically
        the whole point cloud.

        query: (fps_num, 3)
        This is the set of fps samples points.

        k: int
        Number of ref points per query points, i.e.,
        what should be considered the neightbourhood of each
        query point.

    Returns
    -------
        idx: (fps_num, k)
        Returns the indices of k-closest neighbours of each
        query point from the ref points set.
    """

    dist_matrix = jnp.linalg.norm(query[:, None, :] - ref[None, :, :], axis=-1)
    inds = jnp.argsort(dist_matrix, axis=-1)
    return inds[:, :k]

    # NOTE : check out https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    # if the above method is slow.
    # CHECKED And used! about 100x faster than the above method. But doesn't work with vmap :/

    # tree = cKDTree(ref)
    # _, idx = tree.query(query, k=k)
    # return idx


def printParams(
    params: Union[Dict[str, Any], FrozenDict], prefix: str = "params"
) -> None:
    """
    Recursively print the parameters of a model with names and shapes.

    Args
    ----
        params: Union[Dict[str, Any], FrozenDict]
        The parameters of a model.

        prefix: str
        The prefix to be used while printing the parameters.
        Helps to keep track of the nested parameters.

    Returns
    -------
        None
    """
    for key, value in params.items():
        if isinstance(value, dict) or isinstance(value, flax.core.FrozenDict):
            printParams(value, prefix=prefix + '["' + key + '"]')
        else:
            print(f'{prefix}["{key}"]: {value.shape}')

    return


def customTranspose(x: Array) -> Array:
    """
    Transpose the last two axes of an array.
    basically, jnp.transpose(x, (-2, -1)), but this
    is not supported in flax.

    Args
    ----
        x: Array
        The input array.

    Returns
    -------
        reversed_axes_array: Array
        The array with the last two axes transposed.
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


def customSequential(
    x: Array, layers: List[nn.Module], training: bool = False, **kwargs
) -> Array:
    """
    This utility prevents from the overuse of separate flags
    everywhere in the code for training and testing.
    Each module can just call this function.
    """

    for layer in layers:
        if isinstance(layer, nn.BatchNorm):
            x = layer(x, use_running_average=not training)

        elif isinstance(layer, Dropout):
            used_key, kwargs["dropout_key"] = random.split(
                kwargs["dropout_key"]
            )  # in case there are multiple dropout layers
            x = layer(x, deterministic=not training, rng=used_key)

        elif layer == nn.leaky_relu:
            assert (
                "negative_slope" in kwargs
            ), "Leaky ReLU requires negative_slope in kwargs."
            x = layer(x, negative_slope=kwargs["negative_slope"])

        else:
            x = layer(x)

    return x


def sortSelectAndConcat(
    operand: Array, indices: List[List], aux_axis: int = -1, ind_axis: int = 0
) -> Array:
    """
    This function is used during reordering of the features.

    Args
    ----
        operand: Array, (G, C) usually
        The feature tensor that needs to be reordered.

        indices: List[List], List of [(G, 1)] usually
        The indices with which the features need to be reordered.

        aux_axis: int
        The extra axis along which the indices will be repeated.
        NOTE: Might need jnp.tile if there are additional axes
        (> 1).

        ind_axis: int
        The axis along which the sorted features will be concatenated.
        Useful only if len(indices) > 1. This is also the indexed axis.

    Returns
    -------
        ovr_features: Array, (G*len(indices), C)
        The reordered (and concatenated) features tensor.
    """

    ovr_features = []
    for ind in indices:

        feature = jnp.take_along_axis(
            operand,
            jnp.repeat(ind, repeats=operand.shape[-1], axis=aux_axis),
            axis=ind_axis,
        )
        ovr_features.append(feature)

    return jnp.concatenate(ovr_features, axis=ind_axis)


class DropPathV2(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    Minimal reproduction in Jax of the original function used as
    `from timm.models.layers import DropPath`
    """

    drop_prob: float = None

    @nn.compact
    def __call__(self, x: Array, drop_key: KeyArray, training: bool = False) -> Array:
        """
        NOTE: this function drops along the batch dimension.
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        Minimal reproduction in Jax of the original function used as
        `from timm.models.layers import DropPath`

        Args
        ----
            x: Array
            The input tensor.

            drop_key: KeyArray
            The random key for dropping a residual.

            training: bool
            Whether the model is in training mode or not.
            Without training, the drop path will not be applied.

        Returns
        -------
            output: Array
            The output tensor after applying drop path.
        """

        if self.drop_prob == 0.0 or not training:
            return x

        # NOTE :  This is a hack because before vmap I don't have access to the
        # batch dimension. So, I am just taking a key from numpy random and
        # using it as the key for the drop path.
        # completely_random_key = np.random.randint(0, 100000, (1,))
        # dropPath_key = self.make_rng(completely_random_key[0])
        # DOESN'T WORK!
        # Solution: make a batched key object at the start and keep passing it.
            
        if self.drop_prob > 0.0 and drop_key is None:
            raise ValueError("DropPathV2 requires a PRNGKey to be passed.")
        
        if self.drop_prob == 0.0 or not training:
            return x
        
        else:
            keep_prob = 1 - self.drop_prob
            drop = random.uniform(drop_key, (1,), dtype=x.dtype) > keep_prob
            return jnp.where(drop, jnp.zeros_like(x), x / keep_prob)


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.
    Minimal reproduction in Jax of the original function used as
    nn.Identity()
    in PyTorch.

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
        """
        Always returns the input tensor as it is.
        args and kwargs are added to make it compatible with other modules.
        """
        return input


class RMSNorm(nn.Module):
    """
    Taken from https://github.com/radarFudan/mamba-minimal-jax/blob/main/model.py#L344
    as is.
    """

    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args
        ----
            x: Array
            The input tensor.

        Returns
        -------
            output: Array
            The output tensor after applying RMSNorm.
        """

        weight = self.param(
            "weight", nn.initializers.ones, (self.d_model,)
        )  # TODO, maybe use setup will be more clear
        normed = x * jax.lax.rsqrt(
            np.mean(np.square(x), axis=-1, keepdims=True) + self.eps
        )
        output = normed * weight
        return output
