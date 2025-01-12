#!/bin/bash
#SBATCH --job-name=eval_untrained
#SBATCH --output=eval_logs/untrained_%A_%a.out
#SBATCH --error=eval_logs/untrained_%A_%a.err
#SBATCH --partition=alldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-383%16

source ~/.bashrc

CONFIG=$(python3 -c '
import json
import os
with open("eval_configs/untrained_model_configs.json") as f:
    configs = json.load(f)
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(configs[task_id])
')

echo "Running configuration: $CONFIG"
python3 eval.py $CONFIG
