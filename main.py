import sys

sys.path.append(".")

import jax
from jax import random
import jax.numpy as jnp

import os
import json
import numpy as np
import time as time_
from copy import copy
from time import time
from tqdm import tqdm
from typing import TextIO
from flax import jax_utils
from functools import partial
import orbax.checkpoint as ocp
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
    setupDirs,
    trainStep,
    evalStep,
    getIOU,
)

# Constants
log2console = False


def parse_args():
    global log2console

    parser = ArgumentParser()

    # Mamba arguments
    parser.add_argument(
        "--d_model", type=int, default=64, help="Mamba's internal dimension."
    )
    parser.add_argument(
        "--mamba_depth", type=int, default=12, help="Number of Mamba layers to use."
    )
    parser.add_argument(
        "--event_based",
        default=False,
        action="store_true",
        help="Use event based Mamba.",
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
    parser.add_argument("--drop_out", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate.")
    # the original paper also has a drop_path_rate parameter, which is used to initialize
    # a DropPath block but it is never used.

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
    parser.add_argument(
        "--run_name",
        default=None,
        help="Run name. If None, the launching time will be used.",
    )
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
        "--alpha_for_decay", type=float, default=1e-6, help="Alpha for decay."
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        default=False,
        help="Use tracking to w&b.",
    )
    parser.add_argument("--eval_every", type=int, default=10, help="Log every n steps.")

    # One logging arg
    parser.add_argument(
        "--log_to_console", action="store_true", default=False, help="Log to console."
    )

    args = parser.parse_args()
    # Get Mamba arguments
    mamba_args = MambaArgs(
        **{
            arg: getattr(args, arg, None)
            for arg in MambaArgs.__dataclass_fields__.keys()
        }
    )
    mamba_args.conv_bias = not args.no_conv_bias
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
        run_name=args.run_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_points=args.num_points,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha_for_decay=args.alpha_for_decay,
        with_tracking=args.with_tracking,
        eval_every=args.eval_every,
    )

    # Set the logging to console
    log2console = args.log_to_console

    return point_mamba_args, training_args


def printAndLog(to_print, logger: TextIO):
    global log2console
    if log2console:
        print(to_print)
    logger.write(to_print + "\n")
    logger.flush()


def main():

    # Parse arguments
    point_mamba_args, training_args = parse_args()

    # setup directories
    log_file, config_file, checkpoint_dir = setupDirs(run_name=training_args.run_name)

    # Dump all config related stuff into a file
    point_mamba2save = copy(point_mamba_args).__dict__
    point_mamba2save["mamba_args"] = point_mamba2save["mamba_args"].__dict__
    config = {"PointMamba": point_mamba2save, "Training": training_args.__dict__}
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

    # Open the log file
    logger = open(log_file, "w")

    # setup wandb
    if training_args.with_tracking:
        import wandb

        wandb.init(project="jax-pointmamba", name=training_args.run_name, config=config)

    # Print stuff and log it
    args_as_string = "\n".join(
        [f"{k}: {v}" for k, v in point_mamba_args.__dict__.items()]
    )  # because f-strings don't allow '\'
    to_print = f"[*] PointMamba Args: {args_as_string}"
    printAndLog(to_print, logger)
    args_as_string = "\n".join([f"{k}: {v}" for k, v in training_args.__dict__.items()])
    to_print = f"[*] Training Args: {args_as_string}"
    printAndLog(to_print, logger)

    # Create model, optimizer, scheduler and opt_state
    to_print = "[*] Creating model, optimizer, scheduler and opt_state..."
    printAndLog(to_print, logger)
    warmup_steps = 10 * len(ShapenetPartDataset()) // training_args.batch_size
    decay_steps = (
        training_args.num_epochs
        * len(ShapenetPartDataset())
        // training_args.batch_size
    ) - warmup_steps
    model, params, batch_stats, optimizer = getModelAndOpt(
        point_mamba_args,
        16,
        50,
        False,
        "adamw",
        training_args.learning_rate,
        training_args.weight_decay,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        alpha=training_args.alpha_for_decay,
    )

    # Initialize the train state
    to_print = "[*] Creating train state..."
    printAndLog(to_print, logger)
    state = getTrainState(model, params, batch_stats, optimizer)

    # create meta state
    metaData = {
        "epoch": 0,
        "best_instance_avg": 0.0,
        "best_class_avg": 0.0,
        "best_accuracy": 0.0,
        "best_class_avg_accuracy": 0.0,
    }

    # if a checkpoint exists, load it
    init_epoch = 0
    handler = ocp.AsyncCheckpointer(
        ocp.CompositeCheckpointHandler(
            "state",
            "point_mamba_args",
            "training_args",
            "metaData",
        )
    )
    if len(os.listdir(checkpoint_dir)) != 0:
        to_print = "[*] Checkpoint exsits, loading ..."
        printAndLog(to_print, logger)
        ckpt_path = os.path.abspath(
            sorted(os.listdir(checkpoint_dir))[-1]
        )  # get the last checkpoint
        restoredState = handler.restore(os.path.join(checkpoint_dir, ckpt_path))
        # assign the restored variables
        state = restoredState["state"]
        mamba_args = MambaArgs(**restoredState["point_mamba_args"]["mamba_args"])
        point_mamba_args = PointMambaArgs(
            mamba_args=mamba_args,
            mamba_depth=restoredState["point_mamba_args"]["mamba_depth"],
            drop_out=restoredState["point_mamba_args"]["drop_out"],
            drop_path=restoredState["point_mamba_args"]["drop_path"],
            num_group=restoredState["point_mamba_args"]["num_group"],
            group_size=restoredState["point_mamba_args"]["group_size"],
            encoder_channels=restoredState["point_mamba_args"]["encoder_channels"],
            fetch_idx=restoredState["point_mamba_args"]["fetch_idx"],
            leaky_relu_slope=restoredState["point_mamba_args"]["leaky_relu_slope"],
        )
        training_args = TrainingConfig(**restoredState["training_args"])
        init_epoch = restoredState["metaData"]["epoch"]
        metaData["epoch"] = init_epoch
        metaData["best_instance_avg"] = restoredState["metaData"]["best_instance_avg"]
        metaData["best_class_avg"] = restoredState["metaData"]["best_class_avg"]
        metaData["best_accuracy"] = restoredState["metaData"]["best_accuracy"]

    # Get number of devices for distributed training
    num_devices = jax.device_count()
    to_print = f"[*] Number of devices: {num_devices}"
    printAndLog(to_print, logger)

    # Get Train Dataset and Dataloader
    to_print = "[*] Getting Train Dataset and Dataloader..."
    printAndLog(to_print, logger)
    trainval_dataset = ShapenetPartDataset(
        num_points=training_args.num_points, split="trainval", normal_channel=False
    )
    train_bs = training_args.batch_size
    to_print = f"Using batch size: {train_bs}, per device: {int(train_bs/num_devices)}"
    printAndLog(to_print, logger)
    train_dataloader = JAXDataLoader(
        trainval_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
    )

    # Get Test Dataset and Dataloader
    to_print = "[*] Getting Test Dataset and Dataloader..."
    printAndLog(to_print, logger)
    test_dataset = ShapenetPartDataset(
        num_points=training_args.num_points, split="test", normal_channel=False
    )
    test_bs = training_args.batch_size // 2
    to_print = f"Using batch size: {test_bs}, per device: {int(test_bs/num_devices)}"
    printAndLog(to_print, logger)
    test_dataloader = JAXDataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, drop_last=True
    )

    # Initialize the keys
    to_print = "[*] Initializing keys..."
    printAndLog(to_print, logger)
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
    to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Starting training from epoch {init_epoch}..."
    printAndLog(to_print, logger)
    for epoch in range(init_epoch, training_args.num_epochs):

        # Training
        train_loss = 0.0
        ovr_preds = []
        ovr_labels = []
        to_print = f"{'*'*89}\n[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Starting epoch {epoch}...\n{'*'*89}"
        printAndLog(to_print, logger)
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
                shift_key,
                scale_key,
                train_bs,
                num_devices,
                shifter,
                scaler,
                training=True,
                dist=dist,
            )

            # Train step
            state, loss, logits = trainStep_(state, batch)

            # get the overall loss and predictions
            if dist:
                logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
                seg = seg.reshape(-1, seg.shape[-1])
                loss = jnp.sum(loss)

            # Log stuff
            train_loss += loss
            ovr_preds.append(np.array(logits))
            ovr_labels.append(np.array(seg))

        # Logging
        end = time()
        if training_args.with_tracking:
            wandb.log(
                {
                    "train_loss": train_loss / len(trainval_dataset),
                    "train_time": end - start,
                },
                step=epoch,
            )
        to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Epoch: {epoch} - Train Loss: {train_loss/len(trainval_dataset):.4f}, Took {end-start:.2f}s"
        printAndLog(to_print, logger)

        # Evaluate further metrics every eval_every epochs
        if epoch % training_args.eval_every == 0:
            # get metrics
            (
                accuracy,
                class_avg_accuracy,
                class_avg_iou,
                instance_avg_iou,
                shape_ious,
            ) = getIOU(np.concatenate(ovr_preds), np.concatenate(ovr_labels))
            # log it
            to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Epoch: {epoch} - Train Instance Avg: {instance_avg_iou:.4f}, Train Class Avg: {class_avg_iou:.4f}, Train Accuracy: {accuracy:.4f}, Train Class Avg Accuracy: {class_avg_accuracy:.4f}"
            printAndLog(to_print, logger)
            shape_wise = "\n".join(
                f"{key}: {shape_ious[key]:.4f}" for _, key in enumerate(shape_ious)
            )
            to_print = f"[*] Shape-wise IoU:\n{shape_wise}"
            printAndLog(to_print, logger)
            if training_args.with_tracking:
                wandb.log(
                    {
                        "train_instance_avg": instance_avg_iou,
                        "train_class_avg": class_avg_iou,
                        "train_accuracy": accuracy,
                        "train_class_avg_accuracy": class_avg_accuracy,
                    },
                    step=epoch,
                )

        # Evaluation
        eval_loss = 0.0
        ovr_preds = []
        ovr_labels = []
        to_print = f"{'*'*89}\n[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Starting evaluation for epoch {epoch}...\n{'*'*89}"
        printAndLog(to_print, logger)
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
                shift_key,
                scale_key,
                test_bs,
                num_devices,
                shifter,
                scaler,
                training=False,
                dist=dist,
            )

            # Eval step
            cur_loss, logits = evalStep_(state, batch)

            if dist:
                logits = logits.reshape(-1, logits.shape[-2], logits.shape[-1])
                seg = seg.reshape(-1, seg.shape[-1])
                cur_loss = jnp.sum(loss)

            eval_loss += cur_loss
            ovr_preds.append(np.array(logits))
            ovr_labels.append(np.array(seg))

        end = time()
        to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Epoch: {epoch} - Eval Loss: {eval_loss/len(test_dataset):.4f}, Took {end-start:.2f}s"
        printAndLog(to_print, logger)
        # get metrics
        accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou, shape_ious = (
            getIOU(np.concatenate(ovr_preds), np.concatenate(ovr_labels))
        )
        to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Epoch: {epoch} - Eval Instance Avg: {instance_avg_iou:.4f}, Eval Class Avg: {class_avg_iou:.4f}, Eval Accuracy: {accuracy:.4f}, Eval Class Avg Accuracy: {class_avg_accuracy:.4f}"
        printAndLog(to_print, logger)
        shape_wise = "\n".join(
            f"{key}: {shape_ious[key]:.4f}" for _, key in enumerate(shape_ious)
        )
        to_print = f"[*] Shape-wise IoU:\n{shape_wise}"
        printAndLog(to_print, logger)

        # Log to wandb
        if training_args.with_tracking:
            wandb.log(
                {
                    "eval_loss": eval_loss / len(test_dataset),
                    "eval_time": end - start,
                    "eval_instance_avg": instance_avg_iou,
                    "eval_class_avg": class_avg_iou,
                    "eval_accuracy": accuracy,
                    "eval_class_avg_accuracy": class_avg_accuracy,
                },
                step=epoch,
            )

        # update metaData
        metaData["epoch"] = epoch + 1

        # compare with best metrics
        if class_avg_iou > metaData["best_class_avg"]:
            metaData["best_class_avg"] = metaData["best_class_avg"]
        if accuracy > metaData["best_accuracy"]:
            metaData["best_accuracy"] = metaData["best_accuracy"]
        if class_avg_accuracy > metaData["best_class_avg_accuracy"]:
            metaData["best_class_avg_accuracy"] = metaData["best_class_avg_accuracy"]
        if instance_avg_iou > metaData["best_instance_avg"]:
            # save the model
            metaData["best_instance_avg"] = metaData["best_instance_avg"]
            best_path = ocp.test_utils.erase_and_create_empty(
                checkpoint_dir / "best_model"
            )
            point_mamba2save = copy(point_mamba_args).__dict__
            point_mamba2save["mamba_args"] = point_mamba2save["mamba_args"].__dict__
            if dist:
                new_state = jax_utils.unreplicate(state)
            handler.save(
                best_path,
                args=ocp.args.Composite(
                    state=ocp.args.PyTreeSave(new_state),
                    point_mamba_args=ocp.args.JsonSave(point_mamba2save),
                    training_args=ocp.args.JsonSave(training_args.__dict__),
                    metaData=ocp.args.JsonSave(metaData),
                ),
                force=True,
            )
            del new_state

        if epoch % 50 == 0:
            # save the model
            to_print = f"[{time_.strftime('%Y-%m-%d_%H-%M-%S')}] Saving checkpoint..."
            printAndLog(to_print, logger)
            ckpt_path = ocp.test_utils.erase_and_create_empty(
                checkpoint_dir / f"ckpt_{epoch}"
            )
            point_mamba2save = copy(point_mamba_args).__dict__
            point_mamba2save["mamba_args"] = point_mamba2save["mamba_args"].__dict__
            if dist:
                new_state = jax_utils.unreplicate(state)
            handler.save(
                ckpt_path,
                args=ocp.args.Composite(
                    state=ocp.args.PyTreeSave(new_state),
                    point_mamba_args=ocp.args.JsonSave(point_mamba2save),
                    training_args=ocp.args.JsonSave(training_args.__dict__),
                    metaData=ocp.args.JsonSave(metaData),
                ),
                force=True,
            )
            del new_state


if __name__ == "__main__":
    main()
