#!/bin/sh -x

#SBATCH --job-name=port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --hint=multithread
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load the required modules and activate the virtual environment
source ~/.bash_profile
load_v1
activate_jaxpm

python3 main.py --epochs 300 --with_tracking --run_name async_full --d_model 384 --event_based --learning_rate 8e-5 --alpha_for_decay 0.0 --discretize_fn async