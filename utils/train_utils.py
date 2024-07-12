import jax
import optax
import numpy as np
from jax._src import prng
from jax import numpy as jnp
from flax.training import train_state
from typing import Any, Dict, Tuple, List
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


def getModelAndOpt(
    config: PointMambaArgs,
    num_classes: int,
    num_part: int,
    verbose: bool = False,
    opt_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-2,
    decay_steps: int = 1000,
    alpha: float = 0.0,
) -> Tuple[PointMamba, Dict[str, Any], Dict[str, Any], optax.GradientTransformation]:
    # Get the model
    model, params, batch_stats = getModel(config, num_classes, num_part, verbose)

    # Optimizer
    opt = str2opt[opt_name](learning_rate=learning_rate, weight_decay=weight_decay)

    # Scheduler
    sched = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=decay_steps, alpha=alpha
    )

    # Initialize the optimizer and get opt_state
    optimizer = optax.chain(optax.scale_by_schedule(sched), opt)

    return model, params, batch_stats, optimizer  # , opt_state


class TrainState(train_state.TrainState):

    batch_stats: Dict[str, Any]
    # eval_apply_fn: Callable = struct.field(pytree_node=False)


def getTrainState(
    model: PointMamba,
    params: Dict[str, Any],
    # vampped_apply_fn: Callable,
    # eval_apply_fn: Callable,
    batch_stats: Dict[str, Any],
    optimizer: optax.GradientTransformation,
) -> TrainState:

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        # eval_apply_fn=eval_apply_fn,
    )

    return state


def trainStep(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    dist: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    # Define the loss function
    def loss_fn(params):
        (pts, cls_label, fps_keys, droppath_keys, dropout_keys), targets = batch
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

        loss = jnp.sum(
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
        grads = jax.lax.psum(grads, axis_name="data")

    # apply the gradients
    state = state.apply_gradients(grads=grads)
    # Update the batch statistics
    state = state.replace(batch_stats=updates["batch_stats"])

    # get preds
    preds = jnp.argmax(logits, axis=-1)

    return state, loss, preds


def evalStep(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    dist: bool = False,
) -> jnp.ndarray:

    (pts, cls_label, fps_keys, droppath_keys, dropout_keys), seg = batch

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
    loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=seg)
    )

    # get preds
    preds = jnp.argmax(logits, axis=-1)

    return loss, preds


def getIOU(preds: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Computes the Intersection over Union (IoU) for predictions and targets, according to the calculation in the
    PointMamba paper. Source: https://github.com/LMD0311/PointMamba/blob/main/part_segmentation/main.py

    :param preds: predicted labels for each point in the sequence. Shape: [B, N, C]
    :param targets: ground truth labels for each point in the sequence. Shape: [B, N, C]
    """
    categories = []
    ious = []

    for pred, target in zip(preds, targets):
        # Get the category of the item by checking one of the labels in the target
        category = label_to_category[target[0]]

        # Get parts corresponding to the category
        parts = category_to_labels[category]
        num_parts = len(parts)

        # Initialize IoUs for each part
        part_ious = np.zeros((num_parts,))

        # Create masks for target and prediction
        target_masks = (target[:, None] == parts).astype(np.int32)
        pred_masks = (pred[:, None] == parts).astype(np.int32)

        # Calculate IoU for each part
        intersection = np.sum(target_masks & pred_masks, axis=0)
        union = np.sum(target_masks | pred_masks, axis=0)
        part_ious = np.where(union == 0, 1.0, intersection / union)

        categories.append(category)
        ious.append(part_ious.mean())

        instance_avg_iou = np.mean(ious)
        category_avg_iou = np.mean(
            [
                np.mean(ious[categories == category])
                for category in list(category_to_labels.keys())
            ]
        )

    return instance_avg_iou, category_avg_iou
