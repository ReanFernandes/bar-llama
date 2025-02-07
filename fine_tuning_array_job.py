# experiment_manager.py
from itertools import product
import subprocess
import os
import json

# Base paths for Helix
BASE_DIR = "/home/fr/fr_fr/fr_rf1031/bar-llama"
LOGS_DIR = f"{BASE_DIR}/helix-finetuning-logs"

COMPONENTS = {
    'response_formats': [
                    # 'json', 
                    'number_list',
                    'markdown'
                    ],
    'response_types': [
                'answer_first',
                #  'fact_first'
                 ],
    'prompt_types': [
        'few_shot',
        #  'zero_shot'
         ],
    'explanation_types': [
        'structured',
        'unstructured'
         ],
    'seeds': [
        # 'seed_21'
        # 'seed_1337', # with cosine annealling 
        # 'seed_42', # specific qlora based training with batchsize=2 grad-accum 4
        'seed_3991', #qlora, batch size 8 grad accum 2, cosine anneal, 20 epoch
         ],
    'datasets': [
                #  'all_domains_1_samples', 
                #  'all_domains_10_samples', 
                #  'all_domains_20_samples',
                #  'all_domains_75_samples',
                #  'all_domains_125_samples',
                 'all_domains_all_samples'
                 ],
}

def generate_train_configs():
    """Generate all possible train config names"""
    return [
        f"{rf}_{rt}_{pt}_{et}"
        for rf, rt, pt, et in product(
            COMPONENTS['response_formats'],
            COMPONENTS['response_types'],
            COMPONENTS['prompt_types'],
            COMPONENTS['explanation_types']
        )
    ]

def create_array_job():
    train_configs = generate_train_configs()
    
    # Generate all combinations
    configs = [
        f"seeds={seed} dataset={dataset} prompt={train_cfg} train={train_cfg} ++train.training_args.per_device_train_batch_size=8 ++train.training_args.gradient_accumulation_steps=2"
        for seed, dataset, train_cfg in product(
            COMPONENTS['seeds'],
            COMPONENTS['datasets'],
            train_configs
        )
    ]
    #temporary hack to ensure that the same directory is not overriden by the slurm command
    dataset=COMPONENTS['datasets'][0]
    print(f"Total configurations: {len(configs)}")
    print("\nExample configs:")
    for c in configs[:3]:
        print(c)
    
    # Create necessary directories
    os.makedirs(f"{BASE_DIR}/helix-finetuning-logs", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/helix_configs", exist_ok=True)
    
    # Save configs to file
    config_file = f"{BASE_DIR}/helix_configs/train_configs_{dataset}.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Create array job script
    array_script = f"""#!/bin/bash
#SBATCH --job-name=helix-finetuning
#SBATCH --output={LOGS_DIR}/ft_%A_%a.out
#SBATCH --error={LOGS_DIR}/ft_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=39:45:00
#SBATCH --array=0-{len(configs)-1}%12

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
with open("{BASE_DIR}/helix_configs/train_configs_{dataset}.json") as f:
    configs = json.load(f)
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(configs[task_id])
')

echo "Running configuration: $CONFIG"
python3 {BASE_DIR}/train.py $CONFIG
"""
    
    script_path = f"{BASE_DIR}/helix_configs/train_array_{dataset}.sh"
    with open(script_path, 'w') as f:
        f.write(array_script)
    
    print(f"Submitting array job with {len(configs)} tasks...")
    subprocess.run(["sbatch", script_path])
    print("Job submitted!")

if __name__ == "__main__":
    create_array_job()