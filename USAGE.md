# Preparation

## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Python 3.9

## Installation

We recommend using Anaconda for the installation process:
```shell
# Clone the repository
$ git clone https://github.com/LMD0311/PointMamba.git
$ cd PointMamba

# Create virtual env and install PyTorch
$ conda create -n pointmamba python=3.9
$ conda activate pointmamba

# Install basic required packages
(pointmamba) $ pip install -r requirements.txt

# PointNet++
(pointmamba) $ pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Mamba
(pointmamba) $ pip install causal-conv1d==1.1.1
(pointmamba) $ pip install mamba-ssm==1.1.1
```

# Training

## Part Segmentation on ShapeNetPart

```shell
# Training from scratch.
CUDA_VISIBLE_DEVICES=<GPU> python3 main.py --epochs 50 --d_model 64 --with_tracking --run_name _name_
```

Note: Training at d_model=384 is really slow.