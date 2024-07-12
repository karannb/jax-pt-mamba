#!/bin/sh

#SBATCH --job-name=jax-port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

# Load the required modules and activate the virtual environment
source ~/.bash_profile
load_v1
source $ENV_DIR/pointmamba/bin/activate

python3 main.py