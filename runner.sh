#!/bin/sh -x

#SBATCH --job-name=port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --hint=multithread
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load the required modules and activate the virtual environment
source ~/.bash_profile
load_v1
activate_jaxpm

python3 main.py --epochs 50 --with_tracking --run_name major_fixes --d_model 64
