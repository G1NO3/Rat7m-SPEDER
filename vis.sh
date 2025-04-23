#!/bin/bash
#SBATCH --job-name=S_f128_datasets200a200_CD_nonorm1-vis
#SBATCH --output=./sbatch_log/S_f128_datasets200a200_CD_nonorm1-vis.out
#SBATCH --error=./sbatch_log/S_f128_datasets200a200_CD_nonorm1-vis.err
#SBATCH --partition="wu-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"
#SBATCH --mem-per-gpu="8G"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate torch
srun -u python -u visualize.py --alg spedersac --env kms --feature_dim 128 \
                --max_timesteps 1000000 --dir S_f128_datasets200a200_CD_nonorm1 --start_timesteps 200\
                --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cpu --scale_factor 200