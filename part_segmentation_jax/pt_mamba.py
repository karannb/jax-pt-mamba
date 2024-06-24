import jax
from jax import random
import jax.numpy as jnp
from jax.tree_util import Partial

from flax import linen as nn

import torch
import numpy as np
from pointnet2_ops import pointnet2_utils
import pointnet2_utils as pn2_utils
from knn_cuda import KNN
from typing import Optional, Union
from mamba import ResidualBlock, ModelArgs


def create_block(d_model: int,
                 layer_idx: int,
                 d_state: int = 16,
                 expand: int = 2,
                 dt_rank: Union[int, str] = 'auto',
                 d_conv: int = 4,
                 drop_prob: float = 0.,
                 conv_bias: bool = True,
                 bias: bool = False):
    
    model_args = ModelArgs(d_model=d_model,
                           layer_idx=layer_idx,
                           d_state=d_state,
                           expand=expand,
                           dt_rank=dt_rank,
                           d_conv=d_conv,
                           conv_bias=conv_bias,
                           bias=bias)
    
    block = ResidualBlock(args=model_args,
                          drop_prob=drop_prob)
    
    return block


def fps(data: jnp.ndarray,
        number: int) -> jnp.ndarray:
    
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, 'CUDA is not available, fps can only be run on GPU.'
    
    torch_data = torch.from_numpy(np.array(data)).cuda()
    
    fps_idx = pointnet2_utils.furthest_point_sample(torch_data, number)
    fps_data = pointnet2_utils.gather_operation(torch_data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    
    jax_data = jnp.array(fps_data.cpu().numpy())
    
    return jax_data

def my_fps(data: jnp.ndarray,
           number: int) -> jnp.ndarray:
    
    fps_idx = pn2_utils.farthest_point_sample(data, number)
    # fps_data = pn2_utils.gather_operation(torch_data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    
    jax_data = jnp.array(fps_data.cpu().numpy())
    
    return jax_data


# class Group(nn.Module):

if __name__ == "__main__":
    
    # d_model = 4
    # params, block = create_block(d_model=d_model,
    #                              layer_idx=0)
    
    # rand_key, drop_key = random.split(random.key(4))
    # x = random.uniform(rand_key, 
    #                    (2, 3, d_model))
    # out = block.apply(params, x, 
    #                   drop_key=drop_key,
    #                   training=True,)
    
    # print(out)
    
    dummy_data = random.normal(random.PRNGKey(0), 
                               (1, 1024, 3))
    import matplotlib.pyplot as plt
    plt.scatter(dummy_data[0, :, 0], 
                dummy_data[0, :, 1])
    fps_data = fps(dummy_data, 512)
    print(fps_data.shape)
    plt.scatter(fps_data[0, :, 0],
                fps_data[0, :, 1])
    plt.savefig('fps.png')
    plt.close('all')
    
    