#!/bin/sh -x

#SBATCH --job-name=port
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:20:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --hint=multithread
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load the required modules and activate the virtual environment

python3 main.py --epochs 50 --d_model 64 --with_tracking --run_name _name_
