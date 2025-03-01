# Point-Mamba in Jax

This repository holds code for the part-segmentation part of [PointMamba](https://github.com/LMD0311/PointMamba) in JAX & Flax. My [Mamba](https://arxiv.org/abs/2312.00752) implementation borrows significantly from [here](https://github.com/radarFudan/mamba-minimal-jax), a few caveats are that the implementation is much slower than in torch, because of the I/O aware implementation there.
