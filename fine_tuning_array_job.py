# experiment_manager.py
from itertools import product
import subprocess
import os
import json

# 1. Define all your configuration components
COMPONENTS = {
    'response_formats': ['json', 'number_list', 'markdown'],
    'response_types': ['answer_first', 'fact_first'],
    'prompt_types': ['few_shot', 'zero_shot'],
    'explanation_types': ['structured', 'unstructured'],
    'seeds': ['seed_21'], #, 'seed_1337', 'seed_42'],
    'datasets': ['all_domains_all_samples', 
                 'all_domains_1_samples',   
                 'all_domains_10_samples',  
                 'all_domains_20_samples',
                 'all_domains_75_samples',
                 'all_domains_125_samples']
}

def generate_config_names():
    train_configs = [
        f"{rf}_{rt}_{pt}_{et}"
        for rf, rt, pt, et in product(
            COMPONENTS['response_formats'],
            COMPONENTS['response_types'],
            COMPONENTS['prompt_types'],
            COMPONENTS['explanation_types']
        )
    ]
    
    all_configs = [
        f"seeds={seed} dataset={dataset}  train={train}"
        for seed, dataset, train in product(
            COMPONENTS['seeds'],
            COMPONENTS['datasets'],
            train_configs
        )
    ]
    return all_configs

def create_array_job():
    configs = generate_config_names()
    print(f"Total configurations: {len(configs)}")
    print("\nExample configs:")
    for c in configs[:3]:
        print(c)
    print("...\n")
    
    # Create directories
    for dir_name in ["job_configs", "logs", "slurm_scripts"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # Save configs to file
    config_file = "job_configs/experiment_configs.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Create array job script
    array_script = f"""#!/bin/bash
#SBATCH --job-name=ft_array
#SBATCH --output=logs/ft_%A_%a.out
#SBATCH --error=logs/ft_%A_%a.err
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-{len(configs)-1}%12  # Run 4 jobs simultaneously

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
"""
    
    script_path = "slurm_scripts/array_job.sh"
    with open(script_path, 'w') as f:
        f.write(array_script)
    
    print(f"Submitting array job with {len(configs)} tasks...")
    subprocess.run(["sbatch", script_path])
    print("Job submitted!")

if __name__ == "__main__":
    create_array_job()