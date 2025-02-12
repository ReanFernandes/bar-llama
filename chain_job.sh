#!/bin/bash
#SBATCH --job-name=eval_chain
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-inference-logs/ft_%A_%a.out
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-inference-logs/ft_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=1:15:00
#SBATCH --array=0-84%32  # 95 jobs per chain link, max 16 concurrent

export HUGGINGFACE_TOKEN="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"

echo "Chain iteration: ${myloop_counter}"
echo "Job array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

nvidia-smi

module load devel/cuda/12.4
source ~/.bashrc
source /home/fr/fr_fr/fr_rf1031/llama-env/bin/activate

# Calculate the actual config index based on chain iteration and array task ID
config_index=$((($myloop_counter - 1) * 95 + $SLURM_ARRAY_TASK_ID))

# Get the configuration for this job
CONFIG=$(python3 -c "
import json
import os
config_index = $config_index
try:
    with open('/home/fr/fr_fr/fr_rf1031/bar-llama/helix_configs/all_configs.json') as f:
        configs = json.load(f)
    if config_index < len(configs):
        print(configs[config_index])
    else:
        print('NO_MORE_CONFIGS')
except Exception as e:
    print(f'ERROR: {str(e)}')
")

if [ "$CONFIG" != "NO_MORE_CONFIGS" ] && [[ ! "$CONFIG" =~ ^ERROR.* ]]; then
    echo "Running configuration: $CONFIG"
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py $CONFIG
else
    echo "No more configurations to process or error occurred"
    exit 0

# If this is the last array job in this iteration (SLURM_ARRAY_TASK_ID is max)
    if [ "$SLURM_ARRAY_TASK_ID" -eq "84" ]; then
        next_iteration=$((myloop_counter + 1))
        if [ $next_iteration -le 7 ]; then  # 512/40 rounded up = 13 iterations needed
            echo "Submitting next chain iteration ${next_iteration}"
            sbatch --export=ALL,myloop_counter=${next_iteration} /home/fr/fr_fr/fr_rf1031/bar-llama/chain_job.sh
        fi
    fi
fi