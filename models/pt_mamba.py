# jax imports
import jax
from jax import random
import jax.numpy as jnp
from jax.lax import expand_dims

# flax imports
from flax import linen as nn

# other imports
import numpy as np
from functools import partial
from dataclasses import dataclass
from utils.dropout import Dropout
import models.pointnet2_utils as pn2_utils
from models.mamba import ResidualBlock, MambaArgs
from utils.func_utils import (
    RMSNorm,
    knn,
    printParams,
    customTranspose,
    customSequential,
    sortSelectAndConcat,
)

# Type definitions
from jax._src import prng
from jax._src.basearray import Array
from typing import Union, Tuple, Dict, Any, List, Optional

KeyArray = prng.PRNGKeyArray  # Union[Array, ]


@dataclass
class PointMambaArgs:

    mamba_depth: int
    mamba_args: MambaArgs
    drop_out: float = 0.0
    drop_path: float = 0.1
    num_group: int = 128
    group_size: int = 32
    encoder_channels: int = 384
    fetch_idx: Tuple[int] = (3, 7, 11)
    leaky_relu_slope: float = 0.2


def create_block(
    d_model: int,
    event_based: bool = False,
    norm_eps: float = 1e-5,
    rms_norm: bool = False,
    d_state: int = 16,
    expand: int = 2,
    dt_rank: Union[int, str] = "auto",
    d_conv: int = 4,
    conv_bias: bool = True,
    bias: bool = False,
    drop_path: float = 0.1,
) -> ResidualBlock:
    """
    Creates a residual block for the Mamba model.
    NOTE: this is the pre-notm formulation but NOT the
    fused notm one used by the paper which is very fast.

    Args
    ----
        d_model: int
        The model dimension.

        norm_eps: float
        Epsilon value for normalization.

        rms_norm: bool
        Whether to use RMSNorm or not.

        d_state: int
        The state dimension.

        expand: int
        The expansion factor, used before MLP.

        dt_rank: Union[int, str]
        The rank of the delta_t matrix.

        d_conv: int
        The kernel_size of the convolution. (Temporal mixing)

        conv_bias: bool
        Whether to use bias in the convolution.

        bias: bool
        Whether to use bias in the linear layers.

        drop_path: float
        The drop path probability.

    Returns
    -------
        block: ResidualBlock
        The instantiated residual block for the Mamba model.
    """

    model_args = MambaArgs(
        d_model=d_model,
        event_based=event_based,
        rms_norm=rms_norm,
        norm_eps=norm_eps,
        d_state=d_state,
        expand=expand,
        dt_rank=dt_rank,
        d_conv=d_conv,
        conv_bias=conv_bias,
        bias=bias,
    )

    block = ResidualBlock(args=model_args, drop_path=drop_path)

    return block


class Group(nn.Module):

    num_group: int
    group_size: int

    def setup(self):

        self.knn = partial(
            knn, k=self.group_size
        )  # instantiates the knn function with a fixed k

    def __call__(self, pc: Array, key: KeyArray) -> Tuple[Array, Array]:
        """
        Group the points into num_group groups of group_size.

        Args
        ----
            pc: Array, shape=[N, 3]
            The input point cloud.

            key: KeyArray
            The random key for sampling.

        Returns
        -------
            neighborhood: Array, shape=[G, M, 3]
            The grouped points, where G is num_group and M is group_size.

            center: Array, shape=[G, 3]
            The center of each group.
        """

        N, _ = pc.shape

        center = pn2_utils.fps(pc, number=self.num_group, key=key)  # (G, 3)
        idx = self.knn(ref=pc, query=center)  # (G, M)

        assert idx.shape[0] == self.num_group, f"idx.shape[0] : {idx.shape[0]}"
        assert idx.shape[1] == self.group_size, f"idx.shape[1] : {idx.shape[1]}"

        # reshape idx for faster indexing
        idx = idx.reshape(-1)

        # Get the nbrhood
        neighborhood = pc.reshape(N, -1)[idx, :]  # (G*M, 3)
        neighborhood = neighborhood.reshape(
            self.num_group, self.group_size, 3
        )  # (G, M, 3)

        # normalize
        neighborhood = neighborhood - expand_dims(center, dimensions=[1])
        return neighborhood, center


class Encoder(nn.Module):

    encoder_channels: int

    def setup(self):

        self.conv1 = [
            nn.Conv(features=128, kernel_size=(1,), strides=(1,)),
            nn.BatchNorm(axis=-1, axis_name="batch"),
            nn.relu,
            nn.Conv(features=256, kernel_size=(1,), strides=(1,)),
        ]

        self.conv2 = [
            nn.Conv(features=512, kernel_size=(1,), strides=(1,)),
            nn.BatchNorm(axis=-1, axis_name="batch"),
            nn.relu,
            nn.Conv(features=self.encoder_channels, kernel_size=(1,), strides=(1,)),
        ]

    def __call__(self, pc: Array, training: bool = False) -> Array:
        """
        Args
        ----
            pc: Array, shape=[G, M, C]
            The input point cloud.

            training: bool
            Whether the model is in training mode or not.

        Returns
        -------
            global_feature: Array, shape=[G, C]
            The feature vector for each group.
        """

        G, M, _ = pc.shape

        # First convolution
        feature = customSequential(pc, self.conv1, training=training)
        # (G, M, 256)

        # Pick global feature and concatenate to feature
        global_feature = jnp.max(
            feature, axis=1, keepdims=True
        )  # max pooling across all points in a group
        feature = jnp.concatenate([global_feature.repeat(M, axis=1), feature], axis=-1)
        # (G, M, 256) || (G, 1, 256) -> (G, M, 512)

        # Second convolution
        feature = customSequential(feature, self.conv2, training=training)
        # (G, M, encoder_channels)

        # get the feature vector for the center
        global_feature = jnp.max(
            feature, axis=1, keepdims=False
        )  # Max pooling across all points in a group
        return global_feature.reshape(G, -1)  # (G, encoder_channels)


class MixerModelForSegmentation(nn.Module):
    d_model: int
    n_layer: int
    event_based: bool = False
    norm_eps: float = 1e-5
    rms_norm: bool = False
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    drop_path: float = 0.1
    fetch_idx: Tuple[int] = (3, 7, 11)

    # drop_out_in_block: int = 0.
    # residual_in_fp32=False

    def setup(self):

        self.layers = [
            create_block(
                d_model=self.d_model,
                event_based=self.event_based,
                norm_eps=self.norm_eps,
                rms_norm=self.rms_norm,
                d_state=self.d_state,
                expand=self.expand,
                dt_rank=self.dt_rank,
                d_conv=self.d_conv,
                conv_bias=self.conv_bias,
                bias=self.bias,
                drop_path=self.drop_path,
            )
            for _ in range(self.n_layer)
        ]

        self.out_norm = (
            RMSNorm(self.d_model, eps=self.norm_eps)
            if self.rms_norm
            else nn.LayerNorm(epsilon=self.norm_eps)
        )

    def __call__(
        self,
        x: Array,
        pos: Array,
        droppath_key: KeyArray,
        integration_timesteps: Optional[Array] = None,
        training: bool = False,
    ) -> List[Array]:
        """
        Returns the features from the layers specified in fetch_idx.

        Args
        ----
            x: Array
            The input tensor.

            pos: Array
            The positional encoding tensor.

            droppath_key: KeyArray
            The random key for drop path.
            Needs to be passed because randomness is required across vmap.

            training: bool
            Whether the model is in training mode or not.

        Returns
        -------
            features: List
            The list of features from the layers specified in fetch_idx.
        """

        features = []

        hidden_states = x + pos
        residual = None

        for i, layer in enumerate(self.layers):
            # Split the key for drop path in each layer
            used_key, droppath_key = random.split(droppath_key)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                used_key,
                integration_timesteps,
                training=training,
            )
            if i in self.fetch_idx:
                out = self.out_norm(
                    hidden_states + residual if residual is not None else hidden_states
                )
                features.append(out)

        return features


class PointMamba(nn.Module):

    config: PointMambaArgs
    classes: int
    parts: int

    def setup(self):

        assert (
            self.config.encoder_channels == self.config.mamba_args.d_model
        ), f"Encoder channels : {self.config.encoder_channels} and d_model : {self.config.mamba_args.d_model} must be same."

        # Grouper
        self.grouper = Group(
            num_group=self.config.num_group, group_size=self.config.group_size
        )

        # Encoder
        self.encoder = Encoder(encoder_channels=self.config.encoder_channels)

        # Positional Embedding
        self.pos_emb = [
            nn.Dense(128),
            nn.gelu,
            nn.Dense(self.config.mamba_args.d_model),
        ]

        # Mamba model
        self.blocks = MixerModelForSegmentation(
            d_model=self.config.mamba_args.d_model,
            n_layer=self.config.mamba_depth,
            event_based=self.config.mamba_args.event_based,
            rms_norm=self.config.mamba_args.rms_norm,
            fetch_idx=self.config.fetch_idx,
            drop_path=self.config.drop_path,
        )

        # Norm after all Mamba layers
        self.post_norm = nn.LayerNorm()

        # Figure out from paper but is used in Convolution over labels
        self.label_conv = [
            nn.Conv(features=64, kernel_size=(1,), use_bias=False),
            nn.BatchNorm(axis=-1, axis_name="batch"),
            nn.leaky_relu,
        ]

        # Process embedded points using PointNet
        self.propagation_0 = pn2_utils.PointNetFeaturePropagation(
            mlp=[self.config.mamba_args.d_model * 4, 1024]
        )

        # Post-process using simple NN layers
        self.post_layers = [
            nn.Conv(
                features=512, kernel_size=(1,)
            ),  # figure out why 3392 is the number of input channels?
            nn.BatchNorm(axis=-1, axis_name="batch"),
            nn.relu,
            Dropout(0.5),
            nn.Conv(features=256, kernel_size=(1,)),
            nn.BatchNorm(axis=-1, axis_name="batch"),
            nn.relu,
            nn.Conv(features=self.parts, kernel_size=(1,)),
        ]

    def __call__(
        self,
        pts: Array,
        cls_label: Array,
        fps_key: KeyArray,
        droppath_key: KeyArray,
        dropout_key: KeyArray,
        training: bool = False,
    ):
        """
        Run the model on the input point cloud.

        Args
        ----
            pts: Array of shape [3, N]
            The input point cloud.

            cls_label: Array of shape [classes]
            The object being segmented.

            fps_key: KeyArray
            The random key for sampling from the pts.

            droppath_key: KeyArray
            The random key for drop path.
            Needs to be passed because randomness is required across vmap.

            dropout_key: KeyArray
            The random key for dropout.
            Needs to be passed even though randomness IS NOT required across vmap,
            tried a bunch of other things, this is the easiest and most interpretable.

            training: bool
            Whether the model is in training mode or not.

        Returns
        -------
            x: Array of shape [N, classes]
            The output of the model.
        """

        _, N = pts.shape
        pts = customTranspose(pts)  # (N, 3)

        # divide the point cloud in the same form. This is important
        neighborhood, center = self.grouper(pts, fps_key)  # (G, M, 3), (G, 3)
        group_input_tokens = self.encoder(
            neighborhood, training=training
        )  # (G, encoder_channels)

        # Positional encoding
        pos = customSequential(center, self.pos_emb)  # (G, d_model)

        # Reorder, first sort acc to x, then y then z and concatenate
        center_x = center[:, 0].argsort(axis=-1)[:, None]  # (G, 1)
        center_y = center[:, 1].argsort(axis=-1)[:, None]
        center_z = center[:, 2].argsort(axis=-1)[:, None]

        inds = [center_x, center_y, center_z]

        group_input_tokens = sortSelectAndConcat(
            group_input_tokens, inds
        )  # (G*3, encoder_channels)
        pos = sortSelectAndConcat(pos, inds)
        center = sortSelectAndConcat(center, inds)

        # calculate integration timesteps from the centers
        ovr_inds = jnp.concatenate(inds, axis=0)
        timesteps = center[:, 0][ovr_inds]  # get center_x and sort with inds
        integration_timesteps = jnp.diff(timesteps, axis=0, append=timesteps[-1:])

        # Run the Mamba model
        features_list = self.blocks(
            x=group_input_tokens,
            pos=pos,
            droppath_key=droppath_key,
            integration_timesteps=integration_timesteps,
            training=training,
        )
        # (G*3, d_model) * len(fetch_idx)

        features_list = [
            customTranspose(self.post_norm(feature)) for feature in features_list
        ]
        x = jnp.concatenate(features_list, axis=0)  # (d_model*len(fetch_idx), G*3)

        x_max = jnp.max(x, axis=1)  # (d_model*len(fetch_idx)), max_pooling
        x_avg = jnp.mean(x, axis=1)  # mean_pooling
        x_max_feature = expand_dims(x_max, dimensions=[-1]).repeat(repeats=N, axis=-1)
        x_avg_feature = expand_dims(x_avg, dimensions=[-1]).repeat(repeats=N, axis=-1)
        # (d_model*len(fetch_idx), N)

        # Need to tell the model about the class label so it can segment parts
        cls_label_one_hot = cls_label.reshape(1, self.classes)
        # the last reshape is needed because of how nn.Conv operates, expects - (..., features)
        cls_label_one_hot = customSequential(
            x=cls_label_one_hot,
            layers=self.label_conv,
            training=training,
            **{"negative_slope": self.config.leaky_relu_slope},
        )
        cls_label_feature = customTranspose(cls_label_one_hot.repeat(repeats=N, axis=0))
        # (64, N)

        x_global_feature = jnp.concatenate(
            (x_max_feature, x_avg_feature, cls_label_feature), axis=0
        )
        # (d_model*len(fetch_idx)*2 + 64, N)

        # Propagate the RAW points through a PointNet
        f_level_0 = self.propagation_0(
            customTranspose(pts), customTranspose(center), customTranspose(pts), x
        )
        # (G + 3, N)

        # Post-process using simple NN layers
        x = jnp.concatenate(
            (f_level_0, x_global_feature), axis=0
        )  # (d_model*len(fetch_idx)*2 + 64 + G + 3, N)

        x = customSequential(
            x=customTranspose(x),
            layers=self.post_layers,
            training=training,
            **{
                "negative_slope": self.config.leaky_relu_slope,
                "dropout_key": dropout_key,
            },
        )

        return x


# vmap the class
BatchedPointMamba = nn.vmap(
    PointMamba,
    in_axes=(0, 0, 0, 0, 0, None),
    out_axes=0,
    variable_axes={"params": None, "batch_stats": None},
    split_rngs={"params": False},
    axis_name="batch",
)


def getModel(
    config: PointMambaArgs, num_classes: int, num_parts: int, verbose: bool = False
) -> Tuple[PointMamba, Dict[str, Any]]:

    # Keys for init
    input_key, model_key, fps_key, droppath_key, dropout_key = random.split(
        random.PRNGKey(0), 5
    )
    fps_keys = random.split(fps_key, 2)
    droppath_keys = random.split(droppath_key, 2)
    dropout_keys = random.split(dropout_key, 2)
    # Instantiate the model
    model = BatchedPointMamba(config=config, classes=num_classes, parts=num_parts)
    # init args
    dummy_x = random.normal(input_key, (2, 3, 1024))
    dummy_cls = random.randint(
        input_key, (2, 1), minval=0, maxval=num_classes, dtype=jnp.int32
    )
    dummy_cls = jax.nn.one_hot(dummy_cls, num_classes)

    # Initialize the model
    variables = model.init(
        model_key,
        dummy_x,
        dummy_cls,
        fps_keys,
        droppath_keys,
        dropout_keys,
        False,
    )

    params, batch_stats = variables["params"], variables["batch_stats"]

    # Print the model parameters
    if verbose:
        # Print the model parameters
        printParams(params)

    num_params = sum(
        [arr.size for arr in jax.tree.flatten(params)[0] if isinstance(arr, Array)]
    )
    print(f"\nInstantiated Point-Mamba has about {num_params/1e6:.3f}M parameters\n")

    return model, params, batch_stats
