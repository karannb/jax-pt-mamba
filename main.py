import sys

sys.path.append(".")

import jax
from jax import random
import jax.numpy as jnp
from jax import jit, vmap
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

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--rms_norm", type=bool, default=False)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--dt_rank", default="auto")
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--conv_bias", type=bool, default=True)
    parser.add_argument("--bias", type=bool, default=False)
    parser.add_argument("--mamba_depth", type=int, default=3)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--encoder_channels", type=int, default=32)
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
    batch_size = 12
    num_points = 2048
    learning_rate = 0.0002
    weight_decay = 0.05

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
    trainval_dataset = PartNormalDataset(npoints=num_points,
                                         split="trainval",
                                         normal_channel=False)
    train_dataloader = JAXDataLoader(
        trainval_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataset = PartNormalDataset(npoints=num_points,
                                     split="test",
                                     normal_channel=False)
    test_dataloader = JAXDataLoader(test_dataset, batch_size=batch_size)

    # Create model
    model, params, batch_stats = get_model(point_mamba_args, num_cls)

    # Create apply functions for train and eval cases
    # NOTE: this solved 2 issues -
    # 1. because of the static_argnums=6 field, anyways True and False
    # would lead to separate compilation, and this easily allows for
    # that, only eval_apply is used with False.
    # 2. It also allows this mutablility of batch_stats to be separated
    partial_train_apply = partial(model.apply, mutable=["batch_stats"])
    train_apply = jit(vmap(partial_train_apply,
                           in_axes=(None, 0, 0, 0, 0, 0, None),
                           out_axes=(0)),
                      static_argnums=6)
    eval_apply = jit(vmap(model.apply,
                          in_axes=(None, 0, 0, 0, 0, 0, None),
                          out_axes=0),
                     static_argnums=6)

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
    
    # Create training step
    # @jit
    def train_step(
        optimizer: optax.GradientTransformation, opt_state: Any,
        batch: Tuple[jnp.ndarray, jnp.ndarray], params: Any, batch_stats: Any
    ) -> Tuple[optax.GradientTransformation, Any, jnp.ndarray]:

        # Define the loss function
        def loss_fn(params):
            inputs, targets = batch
            (logits, updates) = train_apply(
                {
                    "params": params,
                    "batch_stats": batch_stats
                }, *inputs, True)
            return jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits=logits,
                                            labels=targets)), updates

        # Create a function to compute the gradient
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        # Compute the loss, the gradient and get aux updates
        (loss, updates), grads = grad_fn(params)
        # Update the batch statistics
        batch_stats = updates["batch_stats"]
        # get gradients
        gradients, opt_state = optimizer.update(grads, opt_state, params)
        # take a lr step
        params = optax.apply_updates(params, gradients)

        return optimizer, params, loss

    # Create evaluation step
    # @jit
    def eval_step(params: Any, batch: Tuple[jnp.ndarray,
                                            jnp.ndarray]) -> jnp.ndarray:
        inputs, targets = batch
        logits = eval_apply({
            "params": params,
            "batch_stats": batch_stats
        }, *inputs, False)

        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == targets)

        return acc

    # Initialize the keys
    fps_key, droppath_key, dropout_key, shift_key, scale_key = random.split(
        random.PRNGKey(0), 5)

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            if i == 10:
                break
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
            
            inputs = ((pts, cls_label, fps_keys, droppath_keys, dropout_keys),
                     seg)
            
            # if i < 10:
            start = time()
            optimizer, params, loss = train_step(optimizer, opt_state,
                                                 inputs, params, batch_stats)
            end = time()
            # if i < 2:
            print(f"{i} took {end - start} seconds")
                
            train_loss += loss

        # Evaluation
        test_acc = 0.0
        for i, batch in enumerate(test_dataloader):
            if i == 10:
                break
            (pts, cls_label, seg) = batch
            
            # generate keys
            fps_keys = random.split(fps_key, batch_size + 1)
            fps_key, fps_keys = fps_keys[0], fps_keys[1:]
            droppath_keys = random.split(droppath_key, batch_size + 1)
            droppath_key, droppath_keys = droppath_keys[0], droppath_keys[1:]
            dropout_keys = random.split(dropout_key, batch_size + 1)
            dropout_key, dropout_keys = dropout_keys[0], dropout_keys[1:]
            
            pts = customTranspose(pts)
            cls_label = jax.nn.one_hot(cls_label, num_cls)
            
            inputs = ((pts, cls_label, fps_keys, droppath_keys, dropout_keys),
                     seg)
            
            # if i < 2:
            start = time()
            test_acc += eval_step(params, inputs)
            end = time()
            # if i < 2:
            print(f"{i} took {end - start} seconds")

        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
