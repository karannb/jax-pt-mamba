# jax imports
import jax
from jax import random
import jax.numpy as jnp
from jax.lax import expand_dims
from jax.tree_util import Partial

# flax imports
from flax import linen as nn

# other imports
from dataclasses import dataclass
import part_segmentation_jax.pointnet2_utils as pn2_utils
from part_segmentation_jax.mamba import ResidualBlock, ModelArgs
from part_segmentation_jax.func_utils import (
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
from typing import Union, Tuple, Dict, Any, List
from jaxlib.xla_extension import ArrayImpl, DeviceArray

KeyArray = prng.PRNGKeyArray  # Union[Array, ]


@dataclass
class PointMambaArgs:

    mamba_depth: int
    mamba_args: ModelArgs
    drop_out: float = 0.0
    drop_path: float = 0.1
    num_group: int = 128
    group_size: int = 32
    encoder_channels: int = 384
    fetch_idx: Tuple[int] = (1, 3, 7)
    leaky_relu_slope: float = 0.2


def create_block(
    d_model: int,
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

    model_args = ModelArgs(
        d_model=d_model,
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


def fps(data: Array, number: int, key: KeyArray) -> Array:
    """
    Farthest point sampling algorithm.

    Args
    ----
        data: Array, shape=[N, C] usually
        The input data.

        number: int
        The number of points to sample.

        key: KeyArray
        The random key for sampling.

    Returns
    -------
        fps_data: Array, shape=[number, C]
        The farthest point sampled data.
    """

    fps_idx = pn2_utils.farthest_point_sample(data, number, key)
    fps_data = pn2_utils.index_points(data, fps_idx)

    return fps_data


class Group(nn.Module):

    num_group: int
    group_size: int

    def setup(self):

        self.knn = Partial(
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

        center = fps(pc, number=self.num_group, key=key)  # (G, 3)
        idx = self.knn(ref=pc, query=center)  # (G, M)

        assert type(idx) in [Array, ArrayImpl, DeviceArray], f"idx type : {type(idx)}"
        assert idx.shape[0] == self.num_group, f"idx.shape[1] : {idx.shape[0]}"
        assert idx.shape[1] == self.group_size, f"idx.shape[2] : {idx.shape[1]}"

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
            nn.BatchNorm(axis=-1),
            nn.relu,
            nn.Conv(features=256, kernel_size=(1,), strides=(1,)),
        ]

        self.conv2 = [
            nn.Conv(features=512, kernel_size=(1,), strides=(1,)),
            nn.BatchNorm(axis=-1),
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
        feature = jnp.concatenate([global_feature.repeat(M, axis=1), pc], axis=-1)
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
    norm_eps: float = 1e-5
    rms_norm: bool = False
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    drop_path: float = 0.1
    fetch_idx: Tuple[int] = (1, 3, 7)

    # drop_out_in_block: int = 0.
    # residual_in_fp32=False

    def setup(self):

        self.layers = [
            create_block(
                d_model=self.d_model,
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

    def __call__(self, x: Array, pos: Array, training: bool = False) -> List[Array]:
        """
        Returns the features from the layers specified in fetch_idx.

        Args
        ----
            x: Array
            The input tensor.

            pos: Array
            The positional encoding tensor.

            training: bool
            Whether the model is in training mode or not.

        Returns
        -------
            features: List
            The list of features from the layers specified in fetch_idx.
        """

        features = []

        hidden_states = x + pos

        for i, layer in enumerate(self.layers):
            # drop_key, used_key = random.split(drop_key) drop_key: KeyArray,
            hidden_states = layer(
                hidden_states,
                training=training,
            )
            if (i + 1) in self.fetch_idx:
                features.append(hidden_states)

        hidden_states = self.out_norm(hidden_states)

        return features


class PointMamba(nn.Module):

    classes: int
    config: PointMambaArgs

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
        self.pos_emb = nn.Sequential(
            [nn.Dense(128), nn.gelu, nn.Dense(self.config.mamba_args.d_model)]
        )

        # Mamba model
        self.blocks = MixerModelForSegmentation(
            d_model=self.config.mamba_args.d_model,
            n_layer=self.config.mamba_depth,
            rms_norm=self.config.mamba_args.rms_norm,
            fetch_idx=self.config.fetch_idx,
            drop_path=self.config.drop_path,
        )

        # Norm after all Mamba layers
        self.post_norm = nn.LayerNorm()

        # Figure out from paper but is used in Convolution over labels
        self.label_conv = [
            nn.Conv(features=64, kernel_size=(1,), use_bias=False),
            nn.BatchNorm(axis=-1),
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
            nn.BatchNorm(),
            nn.relu,
            nn.Dropout(0.5),
            nn.Conv(features=256, kernel_size=(1,)),
            nn.BatchNorm(),
            nn.relu,
            nn.Conv(features=self.classes, kernel_size=(1,)),
        ]

    def __call__(
        self, pts: Array, cls_label: Array, fps_key: KeyArray, training: bool = False
    ):
        """
        Run the model on the input point cloud.

        Args
        ----
            pts: Array of shape [B, 3, N]
            The input point cloud.

            cls_label: Array of shape [B, classes]
            The object being segmented.

            fps_key: KeyArray
            The random key for sampling from the pts.

            training: bool
            Whether the model is in training mode or not.

        Returns
        -------
            x: Array of shape [B, N, classes]
            The output of the model.
        """

        B, _, N = pts.shape
        pts = customTranspose(pts)  # B N 3

        # divide the point cloud in the same form. This is important
        neighborhood, center = self.grouper(pts, fps_key)  # (B, G, M, 3), (B, G, 3)
        group_input_tokens = self.encoder(
            neighborhood, training=training
        )  # (B, G, encoder_channels)

        # Positional encoding
        pos = self.pos_emb(center)  # (B, G, d_model)

        # Reorder, first sort acc to x, then y then z and concatenate
        center_x = center[:, :, 0].argsort(axis=-1)[:, :, None]  # (B, G, 1)
        center_y = center[:, :, 1].argsort(axis=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(axis=-1)[:, :, None]

        inds = [center_x, center_y, center_z]

        group_input_tokens = sortSelectAndConcat(
            group_input_tokens, inds
        )  # (B, G*3, encoder_channels)
        pos = sortSelectAndConcat(pos, inds)
        center = sortSelectAndConcat(center, inds)

        # Run the Mamba model
        features_list = self.blocks(x=group_input_tokens, pos=pos, training=training)
        # (B, G*3, d_model) * len(fetch_idx)

        features_list = [
            customTranspose(self.post_norm(feature)) for feature in features_list
        ]
        x = jnp.concatenate(features_list, axis=1)  # (B, d_model*len(fetch_idx), G*3)

        x_max = jnp.max(x, axis=2)  # (B, d_model*len(fetch_idx)), max_pooling
        x_avg = jnp.mean(x, axis=2)  # mean_pooling
        x_max_feature = expand_dims(x_max.reshape(B, -1), dimensions=[-1]).repeat(
            repeats=N, axis=-1
        )
        x_avg_feature = expand_dims(x_avg.reshape(B, -1), dimensions=[-1]).repeat(
            repeats=N, axis=-1
        )
        # (B, d_model*len(fetch_idx), N)

        # Need to tell the model about the class label so it can segment parts
        cls_label_one_hot = cls_label.reshape(B, 1, 16)
        cls_label_one_hot = customSequential(
            x=cls_label_one_hot,
            layers=self.label_conv,
            training=training,
            **{"negative_slope": self.config.leaky_relu_slope},
        )
        cls_label_feature = customTranspose(cls_label_one_hot.repeat(repeats=N, axis=1))
        # (B, 64, N)

        x_global_feature = jnp.concatenate(
            (x_max_feature, x_avg_feature, cls_label_feature), axis=1
        )
        # (B, d_model*len(fetch_idx)*2 + 64, N)

        # Propagate the features through a PointNet
        f_level_0 = self.propagation_0(
            customTranspose(pts), customTranspose(center), customTranspose(pts), x
        )
        # (B, G + 3, N)

        # Post-process using simple NN layers
        x = jnp.concatenate(
            (f_level_0, x_global_feature), axis=1
        )  # (B, d_model*len(fetch_idx)*2 + 64 + G + 3, N)
        x = customSequential(
            x=customTranspose(x),
            layers=self.post_layers,
            training=training,
            **{"negative_slope": self.config.leaky_relu_slope},
        )

        # Final pred
        x = nn.log_softmax(x, axis=1)
        x = customTranspose(x)

        return x


def get_model(
    config: PointMambaArgs, num_classes: int, verbose: bool = False
) -> Tuple[PointMamba, Dict[str, Any]]:

    input_key, model_key, fps_key = random.split(random.PRNGKey(0), 3)
    model = PointMamba(classes=num_classes, config=config)
    dummy_x = random.normal(input_key, (10, 3, 1024))
    dummy_cls = random.randint(
        input_key, (10,), minval=0, maxval=num_classes, dtype=jnp.int32
    )
    dummy_cls = jax.nn.one_hot(dummy_cls, num_classes)
    variables = model.init(
        model_key, pts=dummy_x, fps_key=fps_key, cls_label=dummy_cls, training=False
    )

    if verbose:
        # Print the model parameters
        printParams(variables["params"])

    # Print number of parameters, taken from
    # https://github.com/google/jax/discussions/6153
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    print(f"\nInstantiated Point-Mamba has about {num_params/1e6:.3f}M parameters\n")

    return model, variables


if __name__ == "__main__":

    mamba_conf = ModelArgs(d_model=384)
    new_conf = PointMambaArgs(mamba_depth=2, mamba_args=mamba_conf)
    model, params = get_model(new_conf, num_classes=16)

    fps_key, dropout_key = random.split(random.PRNGKey(0))
    x = jnp.ones((10, 3, 1024))
    cls = jax.nn.one_hot(jnp.ones(10, dtype=jnp.int32), 16)
    out = model.apply(
        params,
        pts=x,
        cls_label=cls,
        fps_key=fps_key,
        training=True,
        rngs={"dropout": dropout_key},
    )
    print(out.shape)
