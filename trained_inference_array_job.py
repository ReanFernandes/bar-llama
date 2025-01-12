# eval_trained_models.py
from itertools import product
import subprocess
import os
import json

COMPONENTS = {
    # Components for constructing eval config names
    'response_formats': ['json', 'number_list', 'markdown'],
    'response_types': ['answer_first', 'fact_first'],
    'prompt_types': ['few_shot', 'zero_shot'],
    'explanation_types': ['structured', 'unstructured'],
    
    # Other components
    'seeds': ['seed_21'],#, 'seed_1337', 'seed_42'
    'datasets': ['all_domains_1_samples', 
                #  'all_domains_10_samples', 
                #  'all_domains_20_samples',
                #  'all_domains_75_samples',
                #  'all_domains_125_samples',
                 'all_domains_all_samples'],
    'generation': ['greedy',
                #    'temp_025',
                #    'temp_06',
                   'temp_09'],
    'evaluation_datasets': ['test_set_1', 
                            'test_set_2'],
    'quantisation': ['full_model', 
                     'quantised_model']
}

def generate_eval_configs():
    """Generate all possible eval config names"""
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
    eval_configs = generate_eval_configs()
    
    # Generate all combinations with eval configs, using + for parameter overrides
    configs = [
        f"seeds={seed} dataset={dataset} generation={gen} evaluation_dataset={eval_set} eval={eval_cfg} ++eval.quantisation_status={quant} ++eval.training_status=trained"
        for seed, dataset, gen, eval_set, quant, eval_cfg in product(
            COMPONENTS['seeds'],
            COMPONENTS['datasets'],
            COMPONENTS['generation'],
            COMPONENTS['evaluation_datasets'],
            COMPONENTS['quantisation'],
            eval_configs
        )
    ]
    
    print(f"Total configurations for trained models: {len(configs)}")
    print("\nExample eval configs:")
    for ec in eval_configs[:3]:
        print(f"Eval config: {ec}")
    print("\nExample full configs:")
    for c in configs[:3]:
        print(c)
    
    # Save configs to file
    os.makedirs("eval_configs", exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)
    
    config_file = "eval_configs/trained_model_configs.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Create array job script
    array_script = f"""#!/bin/bash
#SBATCH --job-name=eval_trained
#SBATCH --output=eval_logs/trained_%A_%a.out
#SBATCH --error=eval_logs/trained_%A_%a.err
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:30
#SBATCH --array=0-{len(configs)-1}%4

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
"""
    
    script_path = "eval_configs/eval_trained_array.sh"
    with open(script_path, 'w') as f:
        f.write(array_script)
    
    print(f"Submitting array job with {len(configs)} tasks...")
    subprocess.run(["sbatch", script_path])
    print("Job submitted!")

if __name__ == "__main__":
    create_array_job()