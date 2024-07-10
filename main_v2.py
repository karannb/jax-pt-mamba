import sys

sys.path.append(".")

import jax
from jax import jit
from jax import random
import flax.linen as nn
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray

import optax
import datetime
import numpy as np
from time import time
from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from typing import Any, Tuple

from models.mamba import MambaArgs
from models.pointnet2_utils import customTranspose
from dataset import PartNormalDataset, JAXDataLoader
from models.pt_mamba import PointMambaArgs, get_model
from utils.provider import jit_batched_random_scale_point_cloud, jit_batched_shift_point_cloud


def parse_args():

    parser = ArgumentParser()

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--rms_norm", type=bool, default=False)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dt_rank", default="auto")
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--conv_bias", type=bool, default=True)
    parser.add_argument("--bias", type=bool, default=False)
    parser.add_argument("--mamba_depth", type=int, default=4)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--fetch_idx", type=tuple, default=(1, 3, 7))
    parser.add_argument("--leaky_relu_slope", type=float, default=0.2)

    args = parser.parse_args()
    mamba_args = MambaArgs(**{
        arg: getattr(args, arg)
        for arg in MambaArgs.__dataclass_fields__.keys()
    })
    point_mamba_args = PointMambaArgs(
        mamba_args=mamba_args,
        mamba_depth=args.mamba_depth,
        drop_out=args.drop_out,
        drop_path=args.drop_path,
        num_group=args.num_group,
        group_size=args.group_size,
        encoder_channels=args.encoder_channels,
        fetch_idx=args.fetch_idx,
        leaky_relu_slope=args.leaky_relu_slope,
    )

    return point_mamba_args


def main():

    # Parse arguments
    point_mamba_args = parse_args()
    print(point_mamba_args)

    # Other params
    num_epochs = 300
    num_cls = 50
    batch_size = 4
    num_points = 100
    learning_rate = 2e-4
    weight_decay = 5e-2

    # Logging
    # timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))

    # Create overall LOG directory
    # exp_dir = Path("./log/")
    # exp_dir.mkdir(exist_ok=True)

    # # Create experiment directory
    # exp_dir = exp_dir.joinpath("part_seg")
    # exp_dir.mkdir(exist_ok=True)

    # # Create experiment sub-directory
    # exp_dir = exp_dir.joinpath(timestr)
    # exp_dir.mkdir(exist_ok=True)

    # # Create sub-directories for checkpoints and logs
    # checkpoints_dir = exp_dir.joinpath("checkpoints/")
    # checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = exp_dir.joinpath("logs/")
    # log_dir.mkdir(exist_ok=True)

    # Get Dataset and DataLoaders
    train_dataset = PartNormalDataset(npoints=num_points,
                                         split="train",
                                         normal_channel=False)
    train_dataloader = JAXDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_dataset = PartNormalDataset(npoints=num_points,
                                    split="val",
                                    normal_channel=False)
    val_dataloader = JAXDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    test_dataset = PartNormalDataset(npoints=num_points,
                                     split="test",
                                     normal_channel=False)
    test_dataloader = JAXDataLoader(test_dataset, batch_size=batch_size,
        drop_last=True)

    # Create model
    model, params, batch_stats = get_model(point_mamba_args, num_cls)

    # Create apply functions for train and eval cases
    # NOTE: this solved 2 issues -
    # 1. because of the static_argnums=6 field, anyways True and False
    # would lead to separate compilation, and this easily allows for
    # that, eval_apply is only used with False.
    # 2. It also allows this mutablility of batch_stats to be separated
    partial_train_apply = partial(model.apply, mutable=['batch_stats'])
    train_apply = nn.vmap(partial_train_apply,
                           in_axes=(None, 0, 0, 0, 0, 0, None),
                           out_axes=(0, None))
                    #   static_argnums=6)
    eval_apply = nn.vmap(model.apply,
                          in_axes=(None, 0, 0, 0, 0, 0, None))
                    # static_argnums=6)
    # Define the parameters
    decay_steps = 1000  # Total number of steps to decay over

    # Create the AdamW optimizer
    adamw_optimizer = optax.adamw(learning_rate=learning_rate,
                                  weight_decay=weight_decay)

    # Define the cosine decay scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=learning_rate,  # Initial learning rate
        decay_steps=decay_steps,  # Total number of steps to decay over
        alpha=0.0  # Minimum learning rate value as a fraction of initial
    )

    # Combine the learning rate schedule with the optimizer
    optimizer = optax.chain(optax.scale_by_schedule(scheduler),
                            adamw_optimizer)
    opt_state = optimizer.init(params)
    
    # Initialize the keys
    fps_key, droppath_key, dropout_key, shift_key, scale_key = random.split(
        random.PRNGKey(0), 5)
    
    def loss(params, batch_stats, inputs, targets):
        logits, updates = train_apply({
            "params": params,
            "batch_stats": batch_stats
        }, *inputs, True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return loss, updates
    
    @jit
    def update(params, batch_stats, opt_state, inputs, targets):
        (loss_val, (batch_updates)), grads = jax.value_and_grad(loss, 
                                                          has_aux=True)(params, batch_stats, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        batch_stats = batch_updates["batch_stats"]
        return loss_val
        

    print("Started Training...")
    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_time = []
        train_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            (pts, cls_label, seg) = batch
            
            # generate keys
            fps_keys = random.split(fps_key, batch_size + 1)
            fps_key, fps_keys = fps_keys[0], fps_keys[1:]
            droppath_keys = random.split(droppath_key, batch_size + 1)
            droppath_key, droppath_keys = droppath_keys[0], droppath_keys[1:]
            dropout_keys = random.split(dropout_key, batch_size + 1)
            dropout_key, dropout_keys = dropout_keys[0], dropout_keys[1:]
            shift_keys = random.split(shift_key, batch_size + 1)
            shift_key, shift_keys = shift_keys[0], shift_keys[1:]
            scale_keys = random.split(scale_key, batch_size + 1)
            scale_key, scale_keys = scale_keys[0], scale_keys[1:]

            # Shift and scale the point cloud
            pts = jit_batched_shift_point_cloud(pts, shift_keys, 0.1)
            pts = jit_batched_random_scale_point_cloud(pts, scale_keys, 0.8,
                                                       1.25)
            pts = customTranspose(pts)
            cls_label = jax.nn.one_hot(cls_label, num_cls)
            batch = ((pts, cls_label, fps_keys, droppath_keys, dropout_keys),
                     seg)
            
            start = time()
            loss_val = update(params, batch_stats, opt_state, *batch)
            end = time()
            train_time += [end - start]
            train_loss += loss_val
            
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss_val}")


if __name__ == "__main__":
    main()
