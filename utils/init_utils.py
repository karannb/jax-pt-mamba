"""
This is needed because Mamba has a particular type of initialization for particular blocks
and layers.
"""

import jax
import math
import flax
from jax import random

def initParams(params, inner_group_size, outer_group_size, n_layer, residuals_per_layer, rand_key):
    
    flat_params = flax.traverse_util.flatten_dict(params)
    for name, param in flat_params.items():
        
        if name[0] == "blocks" and "mixer" in name: # Mamba block w/o norm stuff
            # has convs only but biases become 0.
            if 'bias' in name and 'dt_proj' not in name:
                flat_params[name] = jax.numpy.zeros_like(param)

            elif 'kernel' in name and 'dt_proj' not in name:
                if 'out_proj' in name: # 1.
                    shape = param.shape
                    rand_key, used_key = random.split(rand_key)
                    # make it kaiming uniform
                    fan_in = shape[0]
                    gain = math.sqrt(2 / (math.sqrt(5) ** 2 + 1))
                    # from the paper's implementation,
                    # PyTorch Docs https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
                    value = gain * math.sqrt(3.0 / fan_in)
                    param = random.uniform(used_key, shape, minval=-value, maxval=value)
                    # take num_residuals in account
                    param /= math.sqrt(residuals_per_layer * n_layer)
                    flat_params[name] = param.astype(param.dtype)
                    
                elif 'conv1d' in name:
                    shape = param.shape
                    rand_key, used_key = random.split(rand_key)
                    # sqrt(group/in_feat * kernel_size)
                    # https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
                    value = math.sqrt(inner_group_size / (shape[0] * shape[1]))
                    param = random.uniform(used_key, shape, minval=-value, maxval=value)
                    flat_params[name] = param.astype(param.dtype)
                    
                elif 'in_proj' in name or 'x_proj' in name:
                    shape = param.shape
                    rand_key, used_key = random.split(rand_key)
                    # sqrt(1/in_feat)
                    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                    value = 1 / math.sqrt(shape[0])
                    param = random.uniform(used_key, shape, minval=-value, maxval=value)
                    flat_params[name] = param.astype(param.dtype)
                    
                else:
                    raise ValueError(f"Unknown kernel: {name}")
        
        elif name[0] == "blocks" and ("norm" in name or "out_norm" in name): # Mamba block norm stuff
            
            print(f"Skipping {name}")
        
        elif name[0] == "encoder" or name[0] == "propagation_0" or "post_layers" in name[0] or "label_conv" in name[0]:
            # all use convs
            
            if 'kernel' in name:
                shape = param.shape
                rand_key, used_key = random.split(rand_key)
                # sqrt(group/in_feat * kernel_size)
                # https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
                value = math.sqrt(outer_group_size / (shape[0] * shape[1]))
                param = random.uniform(used_key, shape, minval=-value, maxval=value)
                flat_params[name] = param.astype(param.dtype)
                
                # also initialize the bias if it exists
                # get name
                bias_idx = list(name).index("kernel")
                bias_name = list(name)
                bias_name[bias_idx] = "bias"
                if tuple(bias_name) in flat_params:
                    bias_shape = flat_params[tuple(bias_name)].shape
                    # initialize bias
                    rand_key, used_key = random.split(rand_key)
                    flat_params[tuple(bias_name)] = random.uniform(used_key, bias_shape, minval=-value, maxval=value)
                else:
                    print(f"No bias found for {name}")
                
            elif 'scale' in name: # norm params
                
                print(f"Skipping {name} and its bias, as they are associated with norm layers.")
        
        elif "pos_emb" in name[0]: # now just pos_emb left which is a linear layer
            
            if 'kernel' in name:
                shape = param.shape
                rand_key, used_key = random.split(rand_key)
                # sqrt(1/in_feat)
                # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                value = 1 / math.sqrt(shape[0])
                param = random.uniform(used_key, shape, minval=-value, maxval=value)
                flat_params[name] = param.astype(param.dtype)
                
                # also initialize the bias
                # get name
                bias_idx = list(name).index("kernel")
                bias_name = list(name)
                bias_name[bias_idx] = "bias"
                bias_shape = flat_params[tuple(bias_name)].shape
                # initialize bias
                rand_key, used_key = random.split(rand_key)
                flat_params[tuple(bias_name)] = random.uniform(used_key, bias_shape, minval=-value, maxval=value)
                
            elif 'bias' not in name and 'scale' not in name:
                
                print(f"Skipping {name} and its bias, as they are associated with norm layers.")
            
        else:
            print(f"Skipping {name}")

    # unflatten
    updated_params = flax.traverse_util.unflatten_dict(flat_params)
    
    return updated_params