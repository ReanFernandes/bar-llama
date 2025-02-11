#!/bin/bash
#SBATCH --job-name=helix-finetuning
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs/ft_%A_%a.out
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs/ft_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=00:05:00
#SBATCH --array=0-0%12

# Setup logging
echo "Job array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

# Check GPU status
echo "Checking GPU status..."
nvidia-smi

# Load modules and activate environment
echo "Loading CUDA module..."
module load devel/cuda/12.4

echo "Sourcing .bashrc..."
source ~/.bashrc

echo "Activating llama-env..."
source /home/fr/fr_fr/fr_rf1031/llama-env/bin/activate

# Get the config for this array task
CONFIG=$(python3 -c '
import json
import os
with open("/home/fr/fr_fr/fr_rf1031/bar-llama/helix_configs/train_configs_all_domains_all_samples.json") as f:
    configs = json.load(f)
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(configs[task_id])
')

echo "Running configuration: $CONFIG"
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/train.py $CONFIG
