#!/bin/sh

#SBATCH --job-name=jax-port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

source ~/.bash_profile
source $ENV_DIR/pointmamba/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main.py