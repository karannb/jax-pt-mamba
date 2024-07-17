#!/bin/sh -x

#SBATCH --job-name=port
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --hint=multithread
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load the required modules and activate the virtual environment
source ~/.bash_profile
load_v1
source $ENV_DIR/pointmamba/bin/activate

python3 main.py --epochs 50 --with_tracking --weight_decay 0.00005 --run_name init_debugged_event_based --event_based
