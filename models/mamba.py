# Adapted from https://github.com/radarFudan/mamba-minimal-jax/blob/main/model.py
"""Simple, minimal implementation of Mamba in one file of Jax.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
from typing import Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

from flax import linen as nn

from utils.func_utils import (
    Identity,
    RMSNorm,
    DropPathV2,
    KeyArray,
    Array,
)


@dataclass
class MambaArgs:  # The same as torch version since this does not have any torch specific code
    d_model: int
    norm_eps: float = 1e-5
    rms_norm: bool = False
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class ResidualBlock(nn.Module):
    args: MambaArgs
    drop_path: float = 0.1

    def setup(self):
        """Full Mamba model."""
        super().__init__()
        self.mixer = MambaBlock(self.args)
        self.norm = (
            RMSNorm(self.args.d_model, eps=self.args.norm_eps)
            if self.args.rms_norm
            else nn.LayerNorm(epsilon=self.args.norm_eps)
        )
        self.dropper = (
            DropPathV2(drop_prob=self.drop_path) if self.drop_path > 0.0 else Identity()
        )

    @nn.compact
    def __call__(self, x: Array, drop_key: KeyArray, training=False):
        """
        Args:
            x: shape (l, d)    (See Glossary at top for definitions of b, l, d_in, n...)


        Returns:
            output: shape (l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + self.dropper(
            x, drop_key=drop_key, training=training
        )
        return output


class MambaBlock(nn.Module):
    args: MambaArgs

    def setup(self):
        self.in_proj = nn.Dense(
            features=self.args.d_inner * 2,
            kernel_init=normal(),
            use_bias=self.args.bias,
        )

        # Adjusted for Flax. Flax does not have nn.Conv1d, so you might need to reshape or use a different approach
        self.conv1d = nn.Conv(
            features=self.args.d_inner,
            kernel_size=[self.args.d_conv],
            feature_group_count=self.args.d_inner,
            padding=self.args.d_conv - 1,
            use_bias=self.args.conv_bias,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Dense(
            self.args.dt_rank + self.args.d_state * 2, use_bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Dense(self.args.d_inner, use_bias=True)

        A = jnp.tile(jnp.arange(1, self.args.d_state + 1), (self.args.d_inner, 1))
        self.A_log = self.param(
            "A_log",
            lambda rng, shape: jnp.log(A),
            (self.args.d_inner, self.args.d_state),
        )
        self.D = self.param("D", nn.initializers.ones, (self.args.d_inner,))
        self.out_proj = nn.Dense(
            self.args.d_model, kernel_init=normal(), use_bias=self.args.bias
        )

    def __call__(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (l, 2 * d_in)

        # The split_size is converted to the indices_or_sections method, notice the difference!
        (x, res) = jnp.split(
            x_and_res,
            indices_or_sections=[
                self.args.d_inner,
            ],
            axis=-1,
        )

        # TODO, summarize the difference between torch and jax convolution!
        x = self.conv1d(x)[:l, :]

        x = jax.nn.silu(x)

        y = self.ssm(x)

        y = y * jax.nn.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        # TODO, There is a type conversion to float in the torch version s
        A = -jnp.exp(self.A_log)  # shape (d_in, n)
        D = self.D

        x_dbl = self.x_proj(x)  # (l, dt_rank + 2*n)

        # The split_size is converted to the indices_or_sections method, notice the difference!
        (delta, B, C) = jnp.split(
            x_dbl,
            indices_or_sections=[self.args.dt_rank, self.args.dt_rank + n],
            axis=-1,
        )  # delta: (l, dt_rank). B, C: (l, n)
        delta = jax.nn.softplus(self.dt_proj(delta))  # (l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (l, d_in)
            A: shape (d_in, n)
            B: shape (l, n)
            C: shape (l, n)
            D: shape (d_in,)

        Returns:
            output: shape (l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        deltaA = jnp.exp(jnp.einsum("l d, d n -> l d n", delta, A))
        deltaB_u = jnp.einsum("l d, l n, l d -> l d n", delta, B, u)

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        x = jnp.zeros((d_in, n))
        ys = []
        for i in range(l):
            x = deltaA[i] * x + deltaB_u[i]
            y = jnp.einsum("d n, n -> d", x, C[i, :])
            ys.append(y)
        y = jnp.stack(ys, axis=0)  # shape (l, d_in)
        
        y = y + u * D
        
        return y
        
        # def body_fn(i, carry):
        #     x, ys = carry
        #     x = deltaA[i] * x + deltaB_u[i]
        #     y = jnp.einsum("d n, n -> d", x, C[i, :])
        #     ys = ys.at[i].set(y)
        #     return x, ys

        # x = jnp.zeros((d_in, n))
        # ys = jnp.zeros((l, d_in))
        # x, ys = jax.lax.fori_loop(0, l, body_fn, (x, ys))
        
        # def scan_fn(carry, i):
        #     x, ys = carry
        #     x = deltaA[i] * x + deltaB_u[i]
        #     y = jnp.einsum("d n, n -> d", x, C[i, :])
        #     ys = ys.at[i].set(y)
        #     return (x, ys), None

        # x = jnp.zeros((d_in, n))
        # ys = jnp.zeros((l, d_in))
        # (x, ys), _ = jax.lax.scan(scan_fn, (x, ys), jnp.arange(l))

        # ys = ys + u * D
        # return ys
