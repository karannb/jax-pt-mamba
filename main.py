import sys

sys.path.append(".")

import jax
from jax import random
import jax.numpy as jnp

import numpy as np
from time import time
from tqdm import tqdm
from flax import jax_utils
from functools import partial
from argparse import ArgumentParser

from models.mamba import MambaArgs
from models.pt_mamba import PointMambaArgs
from dataset import ShapenetPartDataset, JAXDataLoader
from utils.augment_utils import (
    batched_random_scale_point_cloud,
    batched_shift_point_cloud,
    jit_batched_random_scale_point_cloud,
    jit_batched_shift_point_cloud,
)
from utils.train_utils import (
    TrainingConfig,
    getModelAndOpt,
    getTrainState,
    prepInputs,
    trainStep,
    evalStep,
    getIOU,
)

def parse_args():

    parser = ArgumentParser()

    # Mamba arguments
    parser.add_argument(
        "--d_model", type=int, default=64, help="Mamba's internal dimension."
    )
    parser.add_argument(
        "--norm_eps", type=float, default=1e-5, help="Epsilon for normalization."
    )
    parser.add_argument(
        "--rms_norm", default=False, action="store_true", help="Use RMSNorm or not."
    )
    parser.add_argument(
        "--d_state", type=int, default=16, help="State dimension inside Mamba."
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=2,
        help="Expansion factor for projection layers, 16 -> 16E.",
    )
    parser.add_argument("--dt_rank", default="auto", help="Rank of the 𝚫t.")
    parser.add_argument(
        "--d_conv", type=int, default=4, help="Kernel size for convolution."
    )
    parser.add_argument(
        "--no_conv_bias",
        action="store_true",
        default=False,
        help="No bias in convolution.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Use bias or not in projection layers.",
    )
    parser.add_argument(
        "--mamba_depth", type=int, default=12, help="Number of Mamba layers to use."
    )
    parser.add_argument("--drop_out", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate.")
    # the original paper also has a drop_path_rate parameter, which is used to initialize
    # a DropPath block but it is never used!

    # PointMamba arguments
    parser.add_argument(
        "--num_group",
        type=int,
        default=128,
        help="Number of groups / number of fps sampled points.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=32,
        help="Size of each group, i.e. num_neighboirs of each fps sampled point.",
    )
    parser.add_argument(
        "--encoder_channels",
        type=int,
        default=64,
        help="Number of channels in the encoder pre-mamba.",
    )
    parser.add_argument(
        "--fetch_idx",
        type=tuple,
        default=(3, 7, 11),
        help="Indices to fetch the features.",
    )
    parser.add_argument(
        "--leaky_relu_slope",
        type=float,
        default=0.2,
        help="Slope for leaky relu activation.",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument(
        "--num_points", type=int, default=2048, help="Number of points."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay."
    )
    parser.add_argument(
        "--alpha_for_decay", type=float, default=0.0, help="Alpha for decay."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        default=False,
        help="Use tracking to w&b.",
    )
    parser.add_argument("--log_every", type=int, default=10, help="Log every n steps.")

    args = parser.parse_args()
    # Get Mamba arguments
    mamba_args = MambaArgs(
        **{arg: getattr(args, arg) for arg in MambaArgs.__dataclass_fields__.keys()}
    )
    # Get PointMamba arguments
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
    # Get Training arguments
    training_args = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_points=args.num_points,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha_for_decay=args.alpha_for_decay,
        with_tracking=args.with_tracking,
        log_every=args.log_every,
    )

    return point_mamba_args, training_args


def main():

    # Parse arguments
    point_mamba_args, training_args = parse_args()
    [print(arg) for arg in point_mamba_args]

    # Get number of devices for distributed training
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")

    # Get Train Dataset and Dataloader
    trainval_dataset = ShapenetPartDataset(
        num_points=training_args.num_points, split="trainval", normal_channel=False
    )
    train_bs = training_args.batch_size
    print(f"Using batch size: {train_bs}, per device: {int(train_bs/num_devices)}")
    train_dataloader = JAXDataLoader(
        trainval_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
    )

    # Get Test Dataset and Dataloader
    test_dataset = ShapenetPartDataset(
        num_points=training_args.num_points, split="test", normal_channel=False
    )
    test_bs = training_args.batch_size / 2
    print(f"Using batch size: {test_bs}, per device: {int(test_bs/num_devices)}")
    test_dataloader = JAXDataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, drop_last=True
    )

    # Create model, optimizer, scheduler and opt_state
    model, params, batch_stats, optimizer = getModelAndOpt(
        point_mamba_args,
        16,
        50,
        False,
        "adamw",
        training_args.learning_rate,
        training_args.weight_decay,
        decay_steps=training_args.num_epochs * len(train_dataloader),
        alpha=0,
    )

    # Initialize the train state
    state = getTrainState(model, params, batch_stats, optimizer)

    # Initialize the keys
    fps_key, droppath_key, dropout_key, shift_key, scale_key = random.split(
        random.PRNGKey(0), 5
    )

    # For multi-device training
    dist = False
    if num_devices > 1:
        dist = True
        state = jax_utils.replicate(state)
        trainStep_ = jax.pmap(partial(trainStep, dist=True), axis_name="device")
        evalStep_ = jax.pmap(evalStep, axis_name="device")
        scaler = jax.pmap(
            batched_random_scale_point_cloud,
            axis_name="device",
            in_axes=(0, 0, None, None),
        )
        shifter = jax.pmap(
            batched_shift_point_cloud, axis_name="device", in_axes=(0, 0, None)
        )
    else:
        trainStep_ = jax.jit(partial(trainStep, dist=False))
        evalStep_ = jax.jit(evalStep)
        scaler = jit_batched_random_scale_point_cloud
        shifter = jit_batched_shift_point_cloud

    # Training loop
    for epoch in range(training_args.num_epochs):
        
        # Training
        train_loss = 0.0
        ovr_preds = []
        ovr_labels = []
        print("*" * 89)
        print("Training...")
        print("*" * 89)
        start = time()
        for inputs in tqdm(train_dataloader, total=len(train_dataloader)):
            (pts, cls_label, seg) = inputs

            # prepare inputs
            batch = prepInputs(
                pts,
                cls_label,
                seg,
                fps_key,
                dropout_key,
                droppath_key,
                train_bs,
                num_devices,
                shifter,
                scaler,
                training=True,
                dist=dist,
            )

            # Train step
            state, loss, preds = trainStep_(state, batch)

            # get the overall loss and predictions
            if dist:
                preds = preds.reshape(num_devices * train_bs, -1)
                seg = seg.reshape(num_devices * train_bs, -1)
                loss = jnp.sum(loss)

            # Log stuff
            train_loss += loss
            ovr_preds.append(np.array(preds))
            ovr_labels.append(np.array(seg))

        # Logging
        end = time()
        print(
            f"Epoch: {epoch}, Loss: {train_loss/len(trainval_dataset):.4f}, Time: {end-start:.2f}s"
        )
        if epoch % 10 == 0:
            # get IOU scores
            instance_avg, category_avg = getIOU(
                np.concatenate(ovr_preds), np.concatenate(ovr_labels)
            )
            print(f"Instance average IoU: {instance_avg:.4f}")
            print(f"Category average IoU: {category_avg:.4f}")
            # get accuracy
            accuracy = np.mean(np.concatenate(ovr_preds) == np.concatenate(ovr_labels))
            print(f"Accuracy: {accuracy*100:.4f}")

        # Evaluation
        ovr_preds = []
        ovr_labels = []
        print("*" * 89)
        print("Evaluating...")
        print("*" * 89)
        start = time()
        for inputs in tqdm(test_dataloader, total=len(test_dataloader)):
            (pts, cls_label, seg) = inputs

            # prepare inputs
            batch = prepInputs(
                pts,
                cls_label,
                seg,
                fps_key,
                dropout_key,
                droppath_key,
                test_bs,
                num_devices,
                shifter,
                scaler,
                training=False,
                dist=dist,
            )

            # Eval step
            cur_loss, preds = evalStep_(state, batch)

            if dist:
                preds = preds.reshape(num_devices * test_bs, -1)
                seg = seg.reshape(num_devices * test_bs, -1)
                cur_loss = jnp.sum(cur_loss)

            loss += cur_loss
            ovr_preds.append(np.array(preds))
            ovr_labels.append(np.array(seg))

        end = time()
        instance_avg, category_avg = getIOU(
            np.concatenate(ovr_preds), np.concatenate(ovr_labels)
        )
        print(f"Instance average IoU: {instance_avg:.4f}")
        print(f"Category average IoU: {category_avg:.4f}")
        # get accuracy
        accuracy = np.mean(np.concatenate(ovr_preds) == np.concatenate(ovr_labels))
        print(f"Accuracy: {accuracy*100:.4f}")
        print(f"Test Loss: {loss/len(test_dataset):.4f}, Took {end-start:.2f}s")


if __name__ == "__main__":
    main()
