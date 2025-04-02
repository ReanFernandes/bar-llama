# eval_untrained_models.py
from itertools import product
import subprocess
import os
import json

COMPONENTS = {
    # Components for constructing eval config names
    'response_formats': [
                        'json', 
                        'number_list', 
                        'markdown'
                        ],
    'response_types': [
                    'answer_first',
                     'fact_first'
                     ],
    'prompt_types': [   
                    'few_shot',
                     'zero_shot'
                     ],
    'explanation_types': [
                    'structured',
                    'unstructured'
                    ],
    
    # Other components (no datasets needed for untrained)
    'seeds': [  
                'seed_21',
                # 'seed_1337', 
                # 'seed_42'
            ],
    'generation': [
                # 'beam_search_10','beam_search_3', 'beam_search_5',
                # 'contrastive_penalty_06_topk_4', 'contrastive_penalty_08_topk_10',
                # 'default_generation',
                # 'stochastic_beam_search_10', 'stochastic_beam_search_3', 'stochastic_beam_search_5',
                'greedy','temp_025', 'temp_06', 'temp_09'],
    'evaluation_datasets': ['test_set_1', 'test_set_2'],
    'quantisation': [
                    'full_model',
                     'quantised_model'
                    ]
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
    
    # Generate all combinations with eval configs, using ++ for overrides
    configs = [
        f"seeds={seed} generation={gen} evaluation_dataset={eval_set} eval={eval_cfg} ++eval.quantisation_status={quant} ++eval.training_status=untrained dataset=untrained"
        for seed, gen, eval_set, quant, eval_cfg in product(
            COMPONENTS['seeds'],
            COMPONENTS['generation'],
            COMPONENTS['evaluation_datasets'],
            COMPONENTS['quantisation'],
            eval_configs
        )
    ]
    
    print(f"Total configurations for untrained models: {len(configs)}")
    print("\nExample eval configs:")
    for ec in eval_configs[:3]:
        print(f"Eval config: {ec}")
    print("\nExample full configs:")
    for c in configs[:3]:
        print(c)
    
    # Save configs to file
    os.makedirs("eval_configs", exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)
    
    config_file = "eval_configs/untrained_model_configs.json"
    with open(config_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    # Create array job script
    array_script = f"""#!/bin/bash
#SBATCH --job-name=eval_untrained
#SBATCH --output={LOGS_DIR}/ft_%A_%a.out
#SBATCH --error={LOGS_DIR}/ft_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=1:00:00
#SBATCH --array=0-{len(configs)-1}%32

export HUGGINGFACE_TOKEN="your_token_here"

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
"""
    
    script_path = "eval_configs/eval_untrained_array.sh"
    with open(script_path, 'w') as f:
        f.write(array_script)
    
    print(f"Submitting array job with {len(configs)} tasks...")
    subprocess.run(["sbatch", script_path])
    print("Job submitted!")

if __name__ == "__main__":
    create_array_job()