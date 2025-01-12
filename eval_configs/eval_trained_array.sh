#!/bin/bash
#SBATCH --job-name=eval_trained
#SBATCH --output=eval_logs/trained_%A_%a.out
#SBATCH --error=eval_logs/trained_%A_%a.err
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:30
#SBATCH --array=0-383%4

source ~/.bashrc

CONFIG=$(python3 -c '
import json
import os
with open("eval_configs/trained_model_configs.json") as f:
    configs = json.load(f)
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(configs[task_id])
')

echo "Running configuration: $CONFIG"
python3 /work/dlclarge1/fernandr-thesis_workspace/bar-llama/eval.py $CONFIG
