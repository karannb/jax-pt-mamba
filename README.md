# Point-Mamba in Jax

This repository holds code for the part-segmentation part of [PointMamba](https://github.com/LMD0311/PointMamba) in JAX & Flax. My [Mamba](https://arxiv.org/abs/2312.00752) implementation borrows significantly from [here](https://github.com/radarFudan/mamba-minimal-jax), a few caveats are that the implementation is much slower than in torch, because of the I/O aware implementation there.

You can use the `runner.sh` file, or run it using
```bash
python3 main.py --epochs 50 --d_model 64 --with_tracking
```

Similar to the original repository, please check [USAGE.md](USAGE.md) and [DATASET.md](DATASET.md) for more details.

This project went ahead with a PyTorch version and in a different direction, the code will be out soon, so this is basically a side project :P.
I had a lot of fun messing around with so much of JAX / Flax.
The paper is here
```bibtex
@misc{schöne2024streamuniversalstatespacemodel,
      title={STREAM: A Universal State-Space Model for Sparse Geometric Data}, 
      author={Mark Schöne and Yash Bhisikar and Karan Bania and Khaleelulla Khan Nazeer and Christian Mayr and Anand Subramoney and David Kappel},
      year={2024},
      eprint={2411.12603},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.12603}, 
}
```
