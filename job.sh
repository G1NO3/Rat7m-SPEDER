#!/bin/bash
#SBATCH --job-name=S_f128_datasets200a200_CD_Yilun_LD_step1e-1_task24_minussignpotential
#SBATCH --output=./sbatch_log/S_f128_datasets200a200_CD_Yilun_LD_step1e-1_task24_minussignpotential.out
#SBATCH --error=./sbatch_log/S_f128_datasets200a200_CD_Yilun_LD_step1e-1_task24_minussignpotential.err
#SBATCH --partition="overcap"
#SBATCH --account="overcap"
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
                --max_timesteps 1000000 --dir S_f128_datasets200a200_CD_Yilun_LD_step1e-1_task24_minussignpotential \
                --eval_freq 1000 --discount 0.9 --batch_size 128 --lasso_coef 0.0 --feature_lr 0.0001
#TODO: change the account from overcap to wu-lab
#SBATCH --account="overcap"
# srun -u python -u visualize.py --alg spedersac --env kms --feature_dim 128 \
#                 --max_timesteps 1000000 --dir S_f128_lasso_0_dataset200_Norm1MLP_doubleoptimized --start_timesteps 200\
#                 --eval_freq 5000 --discount 0.9 --batch_size 5 --times 100 --device cpu --scale_factor 200