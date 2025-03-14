import os
from os.path import join
import jax
import uuid
import flax
import optax
import numpy as np
from jax import random
from jax._src import prng
from jax import numpy as jnp
import orbax.checkpoint as ocp
from dataclasses import dataclass
from utils.func_utils import customTranspose
from utils.dist_utils import reshape_batch_per_device
from utils.dualOptTrainState import DualOptTrainState
from typing import Any, Dict, Tuple, Callable, Optional
from models.pt_mamba import PointMamba, PointMambaArgs, getModel

# type definitions
KeyArray = prng.PRNGKeyArray

category_to_labels = {
    "Airplane": [0, 1, 2, 3],
    "Bag": [4, 5],
    "Cap": [6, 7],
    "Car": [8, 9, 10, 11],
    "Chair": [12, 13, 14, 15],
    "Earphone": [16, 17, 18],
    "Guitar": [19, 20, 21],
    "Knife": [22, 23],
    "Lamp": [24, 25, 26, 27],
    "Laptop": [28, 29],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Mug": [36, 37],
    "Pistol": [38, 39, 40],
    "Rocket": [41, 42, 43],
    "Skateboard": [44, 45, 46],
    "Table": [47, 48, 49],
}
label_to_category = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in category_to_labels.keys():
    for label in category_to_labels[cat]:
        label_to_category[label] = cat

str2opt = {
    "adamw": optax.adamw,
    "sgd": optax.sgd,
    "adam": optax.adam,
}


@dataclass
class TrainingConfig:
    run_name: Optional[str] = None
    batch_size: int = 16
    num_epochs: int = 300
    num_points: int = 2048
    num_workers: int = 0
    learning_rate: float = 0.0002
    weight_decay: float = 5e-4
    alpha_for_decay: float = 0.0
    with_tracking: bool = False
    eval_every: int = 10


def map_params(fn): 
    """
    Convenience function to construct mask_fns,
    there are some parameters that do not need to be re-initialized or
    re-initialized in a specific way, this function allows for that.
    """

    def map_fn(params):
        flat_params = flax.traverse_util.flatten_dict(params)
        flat_mask = {path: fn(path, flat_params[path]) for path in flat_params.keys()}
        return flax.traverse_util.unflatten_dict(flat_mask)

    def inverse_map_fn(params):
        flat_params = flax.traverse_util.flatten_dict(params)
        flat_mask = {
            path: not fn(path, flat_params[path]) for path in flat_params.keys()
        }
        return flax.traverse_util.unflatten_dict(flat_mask)

    return map_fn, inverse_map_fn


def getModelAndOpt(
    config: PointMambaArgs,
    num_classes: int,
    num_part: int,
    verbose: bool = False,
    opt_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-2,
    decay_steps: int = 1000,
    warmup_steps: int = 100,
    const_steps: int = 100,
    alpha: float = 0.0,
) -> Tuple[PointMamba, Dict[str, Any], Dict[str, Any], optax.GradientTransformation]:
    """
    Creates the model and the optimizers
    NOTE: Importantly, if you analyze the code from PointMamba, they perform 2 steps per iteration,
    probably a bug, but I wanted to reproduce the results as is, one of them also clips the gradients.

    Args:
        config: PointMambaArgs, configuration for the model
        num_classes: int, number of classes
        num_part: int, number of parts
        verbose: bool, print the model summary
        opt_name: str, optimizer name
        learning_rate: float, learning rate
        weight_decay: float, weight decay
        decay_steps: int, decay steps for cosine decay
        warmup_steps: int, warmup steps for linear schedule
        const_steps: int, constant steps for constant schedule
        alpha: float, alpha for cosine decay

    Returns:
        Tuple[PointMamba, Dict[str, Any], Dict[str, Any], optax.GradientTransformation] : model, params, batch_stats, optimizer
    """
    # Get the model
    model, params, batch_stats = getModel(config, num_classes, num_part, verbose)

    # Scheduler
    warmup_sched = optax.linear_schedule(
        init_value=alpha, end_value=(0.9) * learning_rate, transition_steps=warmup_steps
    )
    constant_sched = optax.constant_schedule((0.9) * learning_rate)
    cosine_sched = optax.cosine_decay_schedule(
        init_value=(0.9) * learning_rate, decay_steps=decay_steps, alpha=alpha
    )
    boundaries = [warmup_steps, warmup_steps + const_steps]
    sched = optax.join_schedules(
        schedules=[warmup_sched, constant_sched, cosine_sched], boundaries=boundaries
    )

    # Initialize the optimizer and get opt_state
    # keep no weight decay for bias and batch norm params
    no_decay, decay = map_params(
        lambda path, p: (
            path[-1] == "bias" or path[-1] == "A_log" or path[-1] == "D" or p.ndim == 1
        )
    )
    opt1 = optax.chain(
        optax.scale_by_schedule(sched),
        optax.masked(
            str2opt[opt_name](learning_rate=learning_rate, weight_decay=weight_decay),
            mask=decay,
        ),
        optax.masked(
            str2opt[opt_name](learning_rate=learning_rate, weight_decay=0.0),
            mask=no_decay,
        ),
    )

    opt2 = optax.chain(
        optax.clip_by_global_norm(10),
        optax.scale_by_schedule(sched),
        optax.masked(
            str2opt[opt_name](learning_rate=learning_rate, weight_decay=weight_decay),
            mask=decay,
        ),
        optax.masked(
            str2opt[opt_name](learning_rate=learning_rate, weight_decay=0.0),
            mask=no_decay,
        ),
    )

    return model, params, batch_stats, opt1, opt2


def getTrainState(
    model: PointMamba,
    params: Dict[str, Any],
    batch_stats: Dict[str, Any],
    optimizer1: optax.GradientTransformation,
    optimizer2: optax.GradientTransformation,
) -> DualOptTrainState:
    """
    Get the full state for training
    """

    state = DualOptTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx1=optimizer1,
        tx2=optimizer2,
        batch_stats=batch_stats,
    )

    return state


def prepInputs(
    pts,
    cls_label,
    seg,
    fps_key,
    dropout_key,
    droppath_key,
    shift_key,
    scale_key,
    bs,
    num_devices,
    shifter: Callable,
    scaler: Callable,
    training=True,
    dist=False,
):
    """
    Wrapper function to prepare inputs for training and evaluation,
    requires several keys to simulate stateful random operations.

    Args:
        pts: jnp.ndarray, shape=(batch_size, num_points, 3)
        cls_label: jnp.ndarray, shape=(batch_size,)
        seg: jnp.ndarray, shape=(batch_size, num_points)
        fps_key: jnp.ndarray, shape=(2,)
        dropout_key: jnp.ndarray, shape=(2,)
        droppath_key: jnp.ndarray, shape=(2,)
        shift_key: jnp.ndarray, shape=(2,)
        scale_key: jnp.ndarray, shape=(2,)
        bs: int, batch size
        num_devices: int, number of devices
        shifter: Callable, shift the point cloud
        scaler: Callable, scale the point cloud
        training: bool, training or evaluation
        dist: bool, distributed training or not

    Returns:
        Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray] : inputs, targets
    """

    # generate keys
    fps_keys = random.split(fps_key, bs + 1)
    fps_key, fps_keys = fps_keys[0], fps_keys[1:]
    droppath_keys = random.split(droppath_key, bs + 1)
    droppath_key, droppath_keys = droppath_keys[0], droppath_keys[1:]
    dropout_keys = random.split(dropout_key, bs + 1)
    dropout_key, dropout_keys = dropout_keys[0], dropout_keys[1:]
    shift_keys = random.split(shift_key, bs + 1)
    shift_key, shift_keys = shift_keys[0], shift_keys[1:]
    scale_keys = random.split(scale_key, bs + 1)
    scale_key, scale_keys = scale_keys[0], scale_keys[1:]

    # Reshape the batch per device
    if dist:
        pts = reshape_batch_per_device(pts, num_devices)
        cls_label = reshape_batch_per_device(cls_label, num_devices)
        seg = reshape_batch_per_device(seg, num_devices)
        fps_keys = reshape_batch_per_device(fps_keys, num_devices)
        droppath_keys = reshape_batch_per_device(droppath_keys, num_devices)
        dropout_keys = reshape_batch_per_device(dropout_keys, num_devices)
        shift_keys = reshape_batch_per_device(shift_keys, num_devices)
        scale_keys = reshape_batch_per_device(scale_keys, num_devices)

    # Shift and scale the point cloud
    if training:
        pts = shifter(pts, shift_keys, 0.1)
        pts = scaler(pts, scale_keys, 0.8, 1.25)
    pts = customTranspose(pts)
    cls_label = jax.nn.one_hot(cls_label, 16)

    return (
        pts,
        cls_label,
        fps_keys,
        droppath_keys,
        dropout_keys,
    ), seg


def trainStep(
    state: DualOptTrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    dist: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Train the model on a batch of data (supports distributed training)
    """

    # Define the loss function
    def loss_fn(params):
        (
            pts,
            cls_label,
            fps_keys,
            droppath_keys,
            dropout_keys,
        ), targets = batch
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            pts,
            cls_label,
            fps_keys,
            droppath_keys,
            dropout_keys,
            True,
            mutable=["batch_stats"],
        )

        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets
            )
        )

        return loss, (updates, logits)

    # Create a function to compute the gradient
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # Compute the loss, the gradient and get aux updates
    (loss, (updates, logits)), grads = grad_fn(state.params)

    # Average the loss and the gradients
    if dist:
        grads = jax.lax.psum(grads, axis_name="device")

    # apply the gradients
    state = state.apply_gradients(grads=grads)
    # Update the batch statistics
    state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss, logits


def evalStep(
    state: DualOptTrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """
    Evaluate the model on a batch of data
    """

    (
        pts,
        cls_label,
        fps_keys,
        droppath_keys,
        dropout_keys,
    ), seg = batch

    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        pts,
        cls_label,
        fps_keys,
        droppath_keys,
        dropout_keys,
        False,
    )

    # also return the loss
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=seg)
    )

    return loss, logits


def getIOU(
    logits: np.ndarray, targets: np.ndarray
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """
    Compute the accuracy metrics for the model,
    importantly, each category is in a class, and only contrasted with the parts in that class.
    So an airplane wing is not compared to a car wheel.

    Args:
        logits: np.ndarray, shape=(num_samples, num_points, num_classes)
        targets: np.ndarray, shape=(num_samples, num_points)

    Returns:
        Tuple[float, float, float, float, Dict[str, float]] : accuracy, class_avg_accuracy, class_avg_iou, instance_avg_i
    """

    total_seen = 0
    total_correct = 0
    total_seen_class = [0 for _ in range(50)]
    total_correct_class = [0 for _ in range(50)]
    shape_ious = {cat: [] for cat in category_to_labels.keys()}

    for logit, target in zip(logits, targets):

        # get prediciton
        category = label_to_category[target[0]]
        cat_inds = category_to_labels[
            category
        ]  # i.e. get all part labels for this category
        logit = logit[:, cat_inds]  # keep only the parts for this category
        pred = np.argmax(logit, axis=-1) + cat_inds[0]  # get the part label

        # compute accuracy of this pred, target pair
        total_seen += len(target)
        total_correct += np.sum(pred == target)

        # compute class-wise accuracy metrics
        for part in cat_inds:
            total_seen_class[part] += np.sum(target == part)
            total_correct_class[part] += np.sum((pred == part) & (target == part))

        # compute part-wise IoU and average it to get shape IoU
        part_ious = [0.0 for _ in cat_inds]
        for part in cat_inds:
            I = np.sum((pred == part) & (target == part))
            U = np.sum((pred == part) | (target == part))
            if I == 0 and U == 0:
                part_ious[part - cat_inds[0]] = 1.0
            else:
                part_ious[part - cat_inds[0]] = I / U
        shape_ious[category].append(np.mean(part_ious))

    # compute accuracy metrics
    accuracy = total_correct / total_seen
    class_avg_accuracy = np.mean(
        np.array(total_correct_class) / np.array(total_seen_class)
    )

    all_shape_ious = []
    for cat in shape_ious.keys():
        all_shape_ious.extend(shape_ious[cat])
        shape_ious[cat] = np.mean(
            shape_ious[cat]
        )  # average over all shapes in this category

    class_avg_iou = np.mean(list(shape_ious.values()))
    instance_avg_iou = np.mean(all_shape_ious)

    return accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou, shape_ious


def setupDirs(log_dir: str = "ckpts", run_name: Optional[str] = None):

    if run_name == None:
        run_name = uuid.uuid4().hex[:6]  # generate a random run name
        print(f"Run name not provided. Using {run_name} as run name")

    # make the run directory
    os.mkdir(join(log_dir, run_name))
    # log file path
    log_file = join(log_dir, run_name, "ovr.log")
    # config file path
    config_file = join(log_dir, run_name, "config.json")
    # checkpoint directory
    checkpoint_dir = os.path.abspath(join(log_dir, run_name, "checkpoints"))
    # because this is always used for orbax stuff, return a orbax path
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    return log_file, config_file, checkpoint_dir
