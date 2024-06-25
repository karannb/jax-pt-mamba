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
import pointnet2_utils as pn2_utils
from mamba import ResidualBlock, ModelArgs
from func_utils import (
    RMSNorm,
    knn,
    sort_select_and_concat,
    custom_transpose,
    print_params,
)

# Type definitions
from jax._src import prng
from jax._src.basearray import Array
from typing import Union, Tuple, Dict, Any

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
):

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

    fps_idx = pn2_utils.farthest_point_sample(data, number, key)
    fps_data = pn2_utils.index_points(data, fps_idx)

    return fps_data


class Group(nn.Module):

    num_group: int
    group_size: int

    def setup(self):

        self.knn = Partial(knn, k=self.group_size)

    def __call__(self, pc: Array, key: KeyArray) -> Tuple[Array, Array]:

        B, N, _ = pc.shape

        center = fps(pc, self.num_group, key)  # (B, G, 3)
        idx = self.knn(ref=pc, query=center)  # (B, G, M)
        assert idx.shape[1] == self.num_group, f"idx.shape[1] : {idx.shape[1]}"
        assert idx.shape[2] == self.group_size, f"idx.shape[2] : {idx.shape[2]}"

        # translate batch idx steps to directly get the nbrhoods
        idx_base = jnp.arange(0, B).reshape(-1, 1, 1) * N  # (B, 1, 1)
        idx = idx + idx_base
        idx = idx.reshape(-1)

        # Get the nbrhood
        neighborhood = pc.reshape(B * N, -1)[idx, :]
        neighborhood = neighborhood.reshape(B, self.num_group, self.group_size, 3)
        # normalize
        neighborhood = neighborhood - expand_dims(center, dimensions=[2])
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

    def __call__(self, pg: Array, training: bool = False) -> Array:

        B, G, M, C = pg.shape
        pg = pg.reshape(B * G, M, C)

        # First convolution
        for layer in self.conv1:
            if isinstance(layer, nn.BatchNorm):
                pg = layer(pg, use_running_average=not training)
            else:
                pg = layer(pg)
        # (B * G, M, 256)

        # Pick global feature and concatenate to feature
        global_feature = jnp.max(pg, axis=1, keepdims=True)
        feature = jnp.concatenate([global_feature.repeat(M, axis=1), pg], axis=-1)

        # Second convolution
        for layer in self.conv2:
            if isinstance(layer, nn.BatchNorm):
                feature = layer(feature, use_running_average=not training)
            else:
                feature = layer(feature)
        # (B * G, M, 512)

        # get the feature vector for the center
        global_feature = jnp.max(feature, axis=1, keepdims=False)
        return global_feature.reshape(B, G, -1)


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

    def __call__(
        self, x: Array, pos: Array, drop_key: KeyArray, training: bool = False
    ) -> list:

        features = []

        hidden_states = x + pos

        for i, layer in enumerate(self.layers):
            drop_key, used_key = random.split(drop_key)
            hidden_states = layer(hidden_states, drop_key=used_key, training=training)
            if i in self.fetch_idx:
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
        self, pts: Array, cls_label: Array, rand_key: KeyArray, training: bool = False
    ):

        B, C, N = pts.shape
        pts = custom_transpose(pts)  # B N 3
        # divide the point cloud in the same form. This is important
        rand_key, used_key = random.split(rand_key)
        neighborhood, center = self.grouper(pts, used_key)
        group_input_tokens = self.encoder(neighborhood, training=training)  # (B, G, N)

        pos = self.pos_emb(center)

        # Reorder, first sort acc to x, then y then z and concatenate
        center_x = center[:, :, 0].argsort(axis=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(axis=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(axis=-1)[:, :, None]

        inds = [center_x, center_y, center_z]

        group_input_tokens = sort_select_and_concat(group_input_tokens, inds)
        pos = sort_select_and_concat(pos, inds)
        center = sort_select_and_concat(center, inds)

        rand_key, drop_key = random.split(rand_key)
        features_list = self.blocks(
            x=group_input_tokens, pos=pos, drop_key=drop_key, training=training
        )

        feature_list = [
            custom_transpose(self.post_norm(feature)) for feature in features_list
        ]
        x = jnp.concatenate(feature_list, axis=1)  # 1152
        x_max = jnp.max(x, axis=2)
        x_avg = jnp.mean(x, axis=2)
        x_max_feature = expand_dims(x_max.reshape(B, -1), dimensions=[-1]).repeat(
            repeats=N, axis=-1
        )
        x_avg_feature = expand_dims(x_avg.reshape(B, -1), dimensions=[-1]).repeat(
            repeats=N, axis=-1
        )

        # Need to tell the model about the class label so it can segment parts
        cls_label_one_hot = cls_label.reshape(B, 16)
        for layer in self.label_conv:
            if isinstance(layer, nn.BatchNorm):
                cls_label_one_hot = layer(
                    cls_label_one_hot, use_running_average=not training
                )
            elif layer == nn.leaky_relu:
                cls_label_one_hot = layer(
                    cls_label_one_hot, negative_slope=self.config.leaky_relu_slope
                )
            else:
                cls_label_one_hot = layer(cls_label_one_hot)

        cls_label_feature = expand_dims(cls_label_one_hot, dimensions=[-1]).repeat(
            repeats=N, axis=-1
        )
        x_global_feature = jnp.concatenate(
            (x_max_feature, x_avg_feature, cls_label_feature), 1
        )

        f_level_0 = self.propagation_0(
            custom_transpose(pts), custom_transpose(center), custom_transpose(pts), x
        )

        x = jnp.concatenate((f_level_0, x_global_feature), 1)
        for layer in self.post_layers:
            if isinstance(layer, nn.BatchNorm):
                x = layer(x, use_running_average=not training)
            elif isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=not training)
            else:
                x = layer(x)

        # Final pred
        x = nn.log_softmax(x, axis=1)
        x = jnp.transpose(x, (0, 2, 1))

        return x


def get_model(
    config: PointMambaArgs, num_classes: int
) -> Tuple[PointMamba, Dict[str, Any]]:

    input_key, model_key = random.split(random.PRNGKey(0))
    model = PointMamba(classes=num_classes, config=config)
    dummy_x = random.normal(input_key, (10, 3, 1024))
    dummy_cls = random.randint(
        input_key, (10,), minval=0, maxval=num_classes, dtype=jnp.int32
    )
    dummy_cls = jax.nn.one_hot(dummy_cls, num_classes)
    variables = model.init(
        model_key, pts=dummy_x, cls_label=dummy_cls, rand_key=model_key, training=False
    )

    print_params(variables["params"])

    return model, variables


if __name__ == "__main__":

    mamba_conf = ModelArgs(d_model=384)
    new_conf = PointMambaArgs(mamba_depth=2, mamba_args=mamba_conf)
    model, params = get_model(new_conf, num_classes=16)
