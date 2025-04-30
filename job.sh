#!/bin/bash
#SBATCH --job-name=N_f128_actor_autoregressive_actorclone_ctrl
#SBATCH --output=./sbatch_log/N_f128_actor_autoregressive_actorclone_ctrl.out
#SBATCH --error=./sbatch_log/N_f128_actor_autoregressive_actorclone_ctrl.err
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
srun -u python -u main.py --alg spedersac --env kms --feature_dim 128 \
                --max_timesteps 1000000 --dir N_f128_actor_autoregressive_actorclone_ctrl \
                --eval_freq 1000 --discount 0.9 --batch_size 128 --lasso_coef 0.0 --feature_lr 0.0001 \
                --actor_type autoregressive
#TODO: change the account from overcap to wu-lab
#SBATCH --account="overcap"
# srun -u python -u visualize.py --alg spedersac --env kms --feature_dim 128 \
#                 --max_timesteps 1000000 --dir S_f128_lasso_0_dataset200_Norm1MLP_doubleoptimized --start_timesteps 200\
#                 --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cpu --scale_factor 200