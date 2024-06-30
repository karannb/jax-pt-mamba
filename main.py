import sys

sys.path.append(".")

import jax
from jax import random
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax._src.prng import PRNGKeyArray

import optax
import datetime
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from flax import linen as nn
from flax.training.train_state import TrainState

from models.mamba import MambaArgs
from dataset import PartNormalDataset, JAXDataLoader
from models.pt_mamba import PointMambaArgs, get_model
from utils.func_utils import customTranspose
from utils.provider import batched_random_scale_point_cloud, batched_shift_point_cloud


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
    mamba_args = MambaArgs(
        **{arg: getattr(args, arg) for arg in MambaArgs.__dataclass_fields__.keys()}
    )
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


class TrainState(TrainState):
    key: PRNGKeyArray


def main():

    # Parse arguments
    point_mamba_args = parse_args()
    print(point_mamba_args)

    # Other params
    num_epochs = 300
    num_cls = 50
    batch_size = 3
    num_workers = 1
    num_points = 2048
    learning_rate = 0.0002
    weight_decay = 0.05

    # Logging
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    # Create overall LOG directory
    exp_dir = Path("./log/")
    exp_dir.mkdir(exist_ok=True)
    
    # Create experiment directory
    exp_dir = exp_dir.joinpath("part_seg")
    exp_dir.mkdir(exist_ok=True)
    
    # Create experiment sub-directory
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    
    # Create sub-directories for checkpoints and logs
    checkpoints_dir = exp_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    # Get Dataset and DataLoaders
    trainval_dataset = PartNormalDataset(
        npoints=num_points, split="trainval", normal_channel=False
    )
    train_dataloader = JAXDataLoader(
        trainval_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_dataset = PartNormalDataset(
        npoints=num_points, split="test", normal_channel=False
    )
    test_dataloader = JAXDataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # Create model
    model, params = get_model(point_mamba_args, num_cls)
    vmapped_apply = jax.vmap(model.apply, in_axes=(None, 0, 0, 0, 0, None))

    # Create the cosine learning rate schedule
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=num_epochs * len(train_dataloader),
    )

    # Create the AdamW optimizer with the cosine schedule
    optimizer = optax.adamw(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,  # Adjust the weight decay as needed
    )

    # Create state
    train_state = TrainState.create(
        apply_fn=vmapped_apply,
        params=params,
        key=random.PRNGKey(0),
        tx=optimizer,
    )

    # create loss function
    def loss_fn(outputs, targets):

        return optax.softmax_cross_entropy(outputs, targets).mean()

    # Storage for best metrics
    best_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0

    # initialize Keys required
    all_keys = random.split(random.PRNGKey(0), 5)
    fps_key, droppath_key, dropout_key, scale_key, shift_key = all_keys

    for epoch in range(num_epochs):

        loss_batch = []
        num_iter = 0

        for pts, cls_label, targets in train_dataloader:

            num_iter += 1
            # Randomly scale the point cloud
            scale_key, used_key = random.split(scale_key)
            points = batched_random_scale_point_cloud(pts, used_key)

            # Randomly shift the point cloud
            shift_key, used_key = random.split(shift_key)
            points = batched_shift_point_cloud(points, used_key)

            # for input to the model
            points = customTranspose(points)
            cls_label = nn.one_hot(cls_label, num_cls)
            
            # Forward pass
            fps_keys = random.split(fps_key, batch_size + 1)
            fps_key, used_fps_key = fps_keys[0], fps_keys[1:]

            droppath_keys = random.split(droppath_key, batch_size + 1)
            droppath_key, used_droppath_key = droppath_keys[0], droppath_keys[1:]

            dropout_key, used_dropout_key = random.split(dropout_key)

            logits = train_state.apply_fn(
                params,
                points,
                cls_label,
                used_fps_key,
                used_droppath_key,
                True,
                rngs={"dropout": used_dropout_key},
            )
            
            # Compute loss
            loss = loss_fn(logits, targets)
            loss_batch.append(loss)
            
            # Compute gradients
            


if __name__ == "__main__":
    main()
