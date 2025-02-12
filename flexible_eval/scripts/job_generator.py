import json
import os
from itertools import product
import argparse
from pathlib import Path
import sys

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import ExperimentConfig

def generate_eval_configs(components):
    """Generate evaluation configurations from components"""
    return [
        f"{rf}_{rt}_{pt}_{et}"
        for rf, rt, pt, et in product(
            components['response_formats'],
            components['response_types'],
            components['prompt_types'],
            components['explanation_types']
        )
    ]

def generate_all_configs(config: ExperimentConfig):
    """Generate all possible configurations"""
    eval_configs = generate_eval_configs(config.COMPONENTS)
    
    return [
        f"model=llama3 tokenizer=llama3 seeds={seed} dataset={dataset} "
        f"generation={gen} evaluation_dataset={eval_set} eval={eval_cfg} "
        f"++eval.quantisation_status={quant} ++eval.training_status=trained"
        for seed, dataset, gen, eval_set, quant, eval_cfg in product(
            config.COMPONENTS['seeds'],
            config.COMPONENTS['datasets'],
            config.COMPONENTS['generation'],
            config.COMPONENTS['evaluation_datasets'],
            config.COMPONENTS['quantisation'],
            eval_configs
        )
    ]

def create_job_script(config: ExperimentConfig, jobs_per_array: int):
    """Create the job script template"""
    script = f"""#!/bin/bash
#SBATCH --job-name=flex_eval
#SBATCH --output={config.LOGS_DIR}/ft_%A_%a.out
#SBATCH --error={config.LOGS_DIR}/ft_%A_%a.err
#SBATCH --cpus-per-task={config.SLURM_CONFIG['cpus_per_task']}
#SBATCH --partition={config.SLURM_CONFIG['partition']}
#SBATCH --gres=gpu:{config.SLURM_CONFIG['gpus']}
#SBATCH --mem={config.SLURM_CONFIG['mem']}
#SBATCH --time={config.SLURM_CONFIG['time']}
#SBATCH --array=0-{jobs_per_array-1}%{config.SLURM_CONFIG['max_concurrent_jobs']}

export HUGGINGFACE_TOKEN="{config.ENV_SETUP['hf_token']}"

echo "Batch: ${{FLEX_BATCH_ID}}"
echo "Job array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"

nvidia-smi

module load devel/cuda/{config.ENV_SETUP['cuda_version']}
source ~/.bashrc
source {config.ENV_SETUP['env_path']}

# Calculate actual config index
config_index=$((($FLEX_BATCH_ID - 1) * {jobs_per_array} + $SLURM_ARRAY_TASK_ID))

# Get configuration
CONFIG=$(python3 -c "
import json
import os
config_index = $config_index
try:
    with open('${{FLEX_CONFIG_FILE}}') as f:
        configs = json.load(f)
    if config_index < len(configs):
        print(configs[config_index])
    else:
        print('NO_MORE_CONFIGS')
except Exception as e:
    print(f'ERROR: {{str(e)}}')
")

if [ "$CONFIG" != "NO_MORE_CONFIGS" ] && [[ ! "$CONFIG" =~ ^ERROR.* ]]; then
    echo "Running configuration: $CONFIG"
    python3 {config.BASE_DIR}/eval.py $CONFIG
    
    # If this is the last job in the array, submit next batch
    if [ "$SLURM_ARRAY_TASK_ID" -eq "$((${jobs_per_array}-1))" ]; then
        next_batch=$((FLEX_BATCH_ID + 1))
        if [ $next_batch -le $FLEX_TOTAL_BATCHES ]; then
            echo "Submitting next batch ${next_batch}"
            FLEX_BATCH_ID=$next_batch FLEX_CONFIG_FILE=$FLEX_CONFIG_FILE \\
            FLEX_TOTAL_BATCHES=$FLEX_TOTAL_BATCHES \\
            sbatch $FLEX_JOB_SCRIPT
        fi
    fi
else
    echo "No more configurations to process or error occurred"
    exit 0
fi
"""
    return script

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation jobs')
    parser.add_argument('--jobs-per-batch', type=int, default=40,
                       help='Number of jobs per batch')
    parser.add_argument('--start-batch', type=int, default=1,
                       help='Batch number to start with')
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig()
    
    # Generate all configurations
    all_configs = generate_all_configs(config)
    total_configs = len(all_configs)
    
    # Calculate total batches needed
    total_batches = (total_configs + args.jobs_per_batch - 1) // args.jobs_per_batch
    
    print(f"Total configurations: {total_configs}")
    print(f"Jobs per batch: {args.jobs_per_batch}")
    print(f"Total batches needed: {total_batches}")
    
    # Save configurations
    config_file = f"{config.FLEX_DIR}/configs/all_configs.json"
    with open(config_file, 'w') as f:
        json.dump(all_configs, f, indent=2)
    
    # Create job script
    job_script = create_job_script(config, args.jobs_per_batch)
    job_script_path = f"{config.FLEX_DIR}/scripts/job_script.sh"
    with open(job_script_path, 'w') as f:
        f.write(job_script)
    os.chmod(job_script_path, 0o755)
    
    # Submit first batch if requested
    if args.start_batch <= total_batches:
        print(f"\nSubmitting batch {args.start_batch}")
        os.environ.update({
            'FLEX_BATCH_ID': str(args.start_batch),
            'FLEX_CONFIG_FILE': config_file,
            'FLEX_TOTAL_BATCHES': str(total_batches),
            'FLEX_JOB_SCRIPT': job_script_path
        })
        os.system(f"sbatch --export=ALL {job_script_path}")

if __name__ == '__main__':
    main()

