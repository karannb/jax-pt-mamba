import sys

sys.path.append(".")

import jax
from jax import vmap
from jax import random

import optax
import numpy as np
from time import time
from tqdm import tqdm
from functools import partial
from argparse import ArgumentParser

from models.mamba import MambaArgs
from models.pt_mamba import PointMambaArgs
from models.pointnet2_utils import customTranspose
from dataset import ShapenetPartDataset, JAXDataLoader
from utils.provider import (
    jit_batched_random_scale_point_cloud,
    jit_batched_shift_point_cloud,
)
from utils.train_utils import (
    getModelAndOpt,
    getTrainState,
    trainStep,
    evalStep,
    getIOU,
)


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
    parser.add_argument("--mamba_depth", type=int, default=7)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--encoder_channels", type=int, default=128)
    parser.add_argument("--fetch_idx", type=tuple, default=(3, 7))
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


def main():

    # Parse arguments
    point_mamba_args = parse_args()
    print(point_mamba_args)

    # Other params
    num_epochs = 300
    num_cls = 50
    num_points = 100
    learning_rate = 0.0002
    weight_decay = 0.00005

    # Get Dataset and DataLoaders
    trainval_dataset = ShapenetPartDataset(
        num_points=num_points, split="trainval", normal_channel=False
    )
    train_bs = max([i for i in range(1, 16 + 1) if len(trainval_dataset) % i == 0])
    
    train_dataloader = JAXDataLoader(
        trainval_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
    )
    test_dataset = ShapenetPartDataset(
        num_points=num_points, split="test", normal_channel=False
    )
    
    # get the largest divisor of the length of the dataset
    divisors = [i for i in range(1, 16 + 1) if len(test_dataset) % i == 0]
    test_bs = max(divisors) if divisors else None
    test_dataloader = JAXDataLoader(test_dataset, batch_size=test_bs, shuffle=False)

    # Create model, optimizer, scheduler and opt_state
    model, params, batch_stats, optimizer = getModelAndOpt(
        point_mamba_args,
        num_cls,
        False,
        "adamw",
        learning_rate,
        weight_decay,
        decay_steps=1000,
        alpha=0.0,
    )

    # Create apply functions for train and eval cases
    # NOTE: this solved 2 issues -
    # 1. because of the static_argnums=6 field, anyways True and False
    # would lead to separate compilation, and this easily allows for
    # that, only eval_apply is used with False.
    # 2. It also allows this mutablility of batch_stats to be separated
    partial_train_apply = partial(model.apply, mutable=["batch_stats"])
    train_apply = vmap(partial_train_apply, in_axes=(None, 0, 0, 0, 0, 0, None))
    eval_apply = vmap(model.apply, in_axes=(None, 0, 0, 0, 0, 0, None))

    state = getTrainState(params, train_apply, eval_apply, batch_stats, optimizer)

    # Initialize the keys
    fps_key, droppath_key, dropout_key, shift_key, scale_key = random.split(
        random.PRNGKey(0), 5
    )

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        ovr_preds = []
        ovr_labels = []
        print("*"*89)
        print("Training...")
        print("*"*89)
        start = time()
        for inputs in tqdm(train_dataloader, total=len(train_dataloader)):
            (pts, cls_label, seg) = inputs

            # generate keys
            fps_keys = random.split(fps_key, train_bs + 1)
            fps_key, fps_keys = fps_keys[0], fps_keys[1:]
            droppath_keys = random.split(droppath_key, train_bs + 1)
            droppath_key, droppath_keys = droppath_keys[0], droppath_keys[1:]
            dropout_keys = random.split(dropout_key, train_bs + 1)
            dropout_key, dropout_keys = dropout_keys[0], dropout_keys[1:]
            shift_keys = random.split(shift_key, train_bs + 1)
            shift_key, shift_keys = shift_keys[0], shift_keys[1:]
            scale_keys = random.split(scale_key, train_bs + 1)
            scale_key, scale_keys = scale_keys[0], scale_keys[1:]

            # Shift and scale the point cloud
            pts = jit_batched_shift_point_cloud(pts, shift_keys, 0.1)
            pts = jit_batched_random_scale_point_cloud(pts, scale_keys, 0.8, 1.25)
            pts = customTranspose(pts)
            cls_label = jax.nn.one_hot(cls_label, num_cls)

            # Prepare inputs
            batch = ((pts, cls_label, fps_keys, droppath_keys, dropout_keys), seg)

            # Train step
            loss, preds = trainStep(state, batch)

            # Log stuff
            train_loss += loss
            ovr_preds.append(np.array(preds))
            ovr_labels.append(np.array(seg))

        # Logging
        end = time()
        print(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Time: {end-start:.2f}s")
        if epoch % 10 == 0:
            start = time()
            instance_avg, category_avg = getIOU(np.concatenate(ovr_preds), np.concatenate(ovr_labels))
            print(f"Instance average IoU: {instance_avg:.4f}")
            print(f"Category average IoU: {category_avg:.4f}")
            end = time()
            print(f"Time taken for IoU calculation: {end-start:.2f}s")

        # Evaluation
        ovr_preds = []
        ovr_labels = []
        print("*"*89)
        print("Evaluating...")
        print("*"*89)
        start = time()
        for inputs in tqdm(test_dataloader, total=len(test_dataloader)):
            (pts, cls_label, seg) = inputs

            # generate keys
            fps_keys = random.split(fps_key, test_bs + 1)
            fps_key, fps_keys = fps_keys[0], fps_keys[1:]
            droppath_keys = random.split(droppath_key, test_bs + 1)
            droppath_key, droppath_keys = droppath_keys[0], droppath_keys[1:]
            dropout_keys = random.split(dropout_key, test_bs + 1)
            dropout_key, dropout_keys = dropout_keys[0], dropout_keys[1:]
            shift_keys = random.split(shift_key, test_bs + 1)
            shift_key, shift_keys = shift_keys[0], shift_keys[1:]
            scale_keys = random.split(scale_key, test_bs + 1)
            scale_key, scale_keys = scale_keys[0], scale_keys[1:]

            # prepare inputs
            pts = customTranspose(pts)
            cls_label = jax.nn.one_hot(cls_label, num_cls)
            inputs = ((pts, cls_label, fps_keys, droppath_keys, dropout_keys), seg)

            # Eval step
            loss, preds = evalStep(state, inputs)
            ovr_preds.append(np.array(preds))
            ovr_labels.append(np.array(seg))

        end = time()
        instance_avg, category_avg = getIOU(np.concatenate(ovr_preds), np.concatenate(ovr_labels))
        print(f"Instance average IoU: {instance_avg:.4f}")
        print(f"Category average IoU: {category_avg:.4f}")
        print(f"Test Loss: {loss:.4f}, Took {end-start:.2f}s")


if __name__ == "__main__":
    main()
