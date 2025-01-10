#!/bin/bash

#SBATCH --job-name=run-finetuning-one-seed
#SBATCH --output=logs/run-finetuning-one-seed
#SBATCH --error=logs/run-finetuning-one-seed
#SBATCH --cpus-per-task=1
#SBATCH --partition=alldlc2_gpu-l40s
#SBATCH --gres=gpu:1

#SBATCH --time=6:00:00 # 6 hours

# Load the required modules
# run bash rc
source ~/.bashrc

# run train.py, with default config already set in train.py
python3 /work/dlclarge1/fernandr-thesis_workspace/bar-llama/train.py