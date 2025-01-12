"""
This file creates and submits slurm jobs for the fine tuning """

import os
import subprocess
from itertools import product

# Define parameter grids for the sweep
seeds = ["21", "1337", "42"]
datasets = [
    "all_domains_1_samples", "all_domains_10_samples",
    "all_domains_20_samples", "all_domains_75_samples",
    "all_domains_125_samples", "all_domains_all_samples"
]
response_formats = ["json", "number_list", "markdown"]
response_types = ["answer_first", "fact_first"]
prompt_types = ["few_shot", "zero_shot"]
explanation_types = ["structured", "unstructured"]

# Generate all combinations of parameters
trains = [
    f"{rf}_{rt}_{pt}_{et}"
    for rf, rt, pt, et in product(response_formats, response_types, prompt_types, explanation_types)
]

# SLURM script template for job submission
def generate_slurm_script(seed, dataset, train, prompt):
    return f"""#!/bin/bash
#SBATCH --job-name=run-finetuning-one-seed
#SBATCH --output=logs/run-finetuning-one-seed-%j
#SBATCH --error=logs/run-finetuning-one-seed-%j
#SBATCH --cpus-per-task=1
#SBATCH --partition=mldlc2_gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00 # 6 hours
#SBATCH --job-name=ft_{seed}_{dataset}_{train}

source ~/.bashrc
conda activate thesis_env
module load devel/cuda

echo "Running fine-tuning with Seed={seed}, Dataset={dataset}, Train Config={train}, Prompt={prompt}"
python3 train_llama.py seed={seed} dataset={dataset} train={train} prompt={prompt}
"""

# Function to submit jobs
def submit_jobs():
    job_count = 0
    os.makedirs("logs", exist_ok=True)  # Ensure logs directory exists
    os.makedirs("slurm_scripts", exist_ok=True)  # Ensure slurm_scripts directory exists

    for seed, dataset, train in product(seeds, datasets, trains):
        prompt = train  # Tie prompt to the train configuration

        # Generate the SLURM script content
        slurm_script = generate_slurm_script(seed, dataset, train, prompt)

        # Save the script to a temporary file
        script_path = f"slurm_scripts/ft_{job_count}.sh"
        with open(script_path, "w") as script_file:
            script_file.write(slurm_script)

        # Submit the script using sbatch
        print(f"Submitting job: Seed={seed}, Dataset={dataset}, Train Config={train}, Prompt={prompt}")
        subprocess.run(["sbatch", script_path])
        job_count += 1

    print(f"Total jobs submitted: {job_count}")

# Main function
if __name__ == "__main__":
    print("Starting job submission...")
    submit_jobs()
    print("All jobs submitted.")
