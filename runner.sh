#!/bin/sh

#SBATCH --job-name=jax-port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --time=05:59:59
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

source ~/.bash_profile
source $ENV_DIR/pointmamba/bin/activate

python3 main.py