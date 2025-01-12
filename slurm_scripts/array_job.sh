#!/bin/bash
#SBATCH --job-name=ft_array
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-143%12  # Run 4 jobs simultaneously

source ~/.bashrc

CONFIG=$(python3 -c '
import json
import os
with open("job_configs/experiment_configs.json") as f:
    configs = json.load(f)
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(configs[task_id])
')

echo "Running configuration: $CONFIG"
python3 train.py $CONFIG
