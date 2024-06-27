import jax
from jax import random
import jax.numpy as jnp
from jax._src import prng
from jax._src.basearray import Array

import os, threading
import numpy as np
import flax.linen as nn

from typing import Union, Tuple

KeyArray = Union[Array, prng.PRNGKeyArray]


def pc_normalize(pc: Array) -> Array:
    """
    Normalize a point cloud.

    Input:
        pc: point cloud data, [N, 3]

    Output:
        normalized_pc: normalized point cloud data, [N, 3]
    """
    centroid = jnp.mean(pc, axis=0)
    pc = pc - centroid
    m = jnp.max(jnp.sqrt(jnp.sum(pc**2, axis=1)))
    normalized_pc = pc / m
    return normalized_pc


def square_distance(src: Array, dst: Array) -> Array:
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * jnp.matmul(src, jnp.transpose(dst, (0, 2, 1)))
    dist += jnp.sum(src**2, -1).reshape(B, N, 1)
    dist += jnp.sum(dst**2, -1).reshape(B, 1, M)
    return dist


def index_points(points: Array, idx: Array) -> Array:
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [S, ]
    Return:
        new_points:, indexed points data, [S, C]
    """
    # NOTE: this isn't needed if everything is 1D
    # But, I have kept it around not to change too
    # much of the code.

    new_points = points[idx, :]
    return new_points


def farthest_point_sample(xyz: Array, fps_key: KeyArray, npoint: int) -> Array:
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """

    N, C = xyz.shape

    # Initialize the centroids as the zeroth point
    centroids = jnp.zeros((npoint,), dtype=jnp.int32)

    # Set the distance to each point as infinity (very large number)
    distance = jnp.ones((N,)) * 1e10

    # Randomly select the farthest point, the [0] is somehow necessary
    # FIXME: I don't know why this is necessary
    farthest = jax.random.randint(fps_key, (1,), 0, N)[0]

    def body_fn(i: int, val: Tuple[Array, Array, Array]):
        centroids, distance, farthest = val

        # Set a new centroid as the farthest point
        centroids = centroids.at[i].set(farthest)

        # Get the coordinates of the current centroid
        centroid = xyz[farthest, :].reshape(1, C)

        # Calculate the Euclidean distance between the new centroid and all other points
        dist = jnp.sum((xyz - centroid) ** 2, axis=-1)

        # Update the distance to the nearest centroid for each point
        distance = jnp.where(dist < distance, dist, distance)

        # Get the index of the farthest point from each centroid
        farthest = jnp.argmax(distance, axis=-1)

        return centroids, distance, farthest

    centroids, _, _ = jax.lax.fori_loop(
        0, npoint, body_fn, (centroids, distance, farthest)
    )
    return centroids


class PointNetFeaturePropagation(nn.Module):

    mlp: list

    @nn.compact
    def __call__(
        self,
        xyz1: Array,
        xyz2: Array,
        points1: Array,
        points2: Array,
        training: bool = False,
    ) -> Array:
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = jnp.transpose(xyz1, (0, 2, 1))
        xyz2 = jnp.transpose(xyz2, (0, 2, 1))

        points2 = jnp.transpose(points2, (0, 2, 1))
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(N, axis=1)
        else:
            dists = square_distance(xyz1, xyz2)
            new_dists, idx = dists.sort(axis=-1), dists.argsort(axis=-1)
            dists, idx = new_dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = jnp.sum(dist_recip, axis=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = jnp.sum(
                index_points(points2, idx) * weight.reshape(B, N, 3, 1), axis=2
            )

        if points1 is not None:
            points1 = jnp.transpose(points1, (0, 2, 1))
            new_points = jnp.concatenate([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        new_points = jnp.transpose(new_points, (0, 2, 1))
        for out_channel in self.mlp:
            new_points = nn.relu(
                nn.BatchNorm(axis=-1, use_running_average=not training)(
                    nn.Conv(out_channel, (1,))(new_points)
                )
            )

        return new_points
