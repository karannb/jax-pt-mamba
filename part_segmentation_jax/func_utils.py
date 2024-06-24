import jax
from jax import random
import jax.numpy as jnp

import numpy as np
from flax import linen as nn


def drop_path(x:jnp.ndarray, drop_prob: float = 0., key: random.PRNGKey = None, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + random.uniform(key, shape, dtype=x.dtype)
    random_tensor = jnp.floor(random_tensor)  # binarize
    output = jnp.divide(x, keep_prob) * random_tensor
    return output


class DropPathV2(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
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
    def __call__(self, input: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return input
    

class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.d_model,)) # TODO, maybe use setup will be more clear
        normed = x * jax.lax.rsqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps)
        output = normed * weight
        return output