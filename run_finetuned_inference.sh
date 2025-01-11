#!/bin/bash

#SBATCH --job-name=run-inference-one-seed
#SBATCH --output=logs/run-inference-one-seed
#SBATCH --error=logs/run-inference-one-seed
#SBATCH --cpus-per-task=1
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1

#SBATCH --time=1:00:00 # 1 hours

# Load the required modules
# run bash rc
source ~/.bashrc

# run train.py, with default config already set in train.py
python3 /work/dlclarge1/fernandr-thesis_workspace/bar-llama/eval.py