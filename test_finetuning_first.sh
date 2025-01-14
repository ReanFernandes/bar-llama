#!/bin/bash

#SBATCH --job-name=helix-finetuning-one-seed
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs/helix-finetuning-one-seed-%j
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs/helix-finetuning-one-seed-%j
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G                         # Memory per job
#SBATCH --time=6:00:00 # 6 hours

# Ensure logs directory exists
echo "Ensuring logs directory exists..."
mkdir -p /home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs
echo "Checking GPU status with nvidia-smi..."
nvidia-smi
# Load necessary modules or activate your environment
echo "Loading CUDA module version 12.4..."
module load devel/cuda/12.4
echo "Sourcing .bashrc..."
source ~/.bashrc
echo "Activating llama-env..."
source llama-env/bin/activate

# run train.py, with default config already set in train.py
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/train.py