#!/bin/bash

#SBATCH --job-name=helix-inference-one-seed
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-inference-logs/helix-finetuning-one-seed-%j
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/helix-inference-logs/helix-finetuning-one-seed-%j
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G                         # Memory per job
#SBATCH --time=10:00:00 # 10 hours

# Ensure logs directory exists
echo "Ensuring logs directory exists..."
mkdir -p /home/fr/fr_fr/fr_rf1031/bar-llama/helix-finetuning-logs
echo "Checking GPU status with nvidia-smi..."
nvidia-smi
# Load necessary modules or activate your environment
echo "Loading CUDA module version 12.4..."
module load devel/cuda/12.4
echo "Sourcing .bashrc..."
source ~/.bashrc
echo "Activating llama-env..."
source llama-env/bin/activate

# run train.py, with default config already set in train.py
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_09 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py model=llama2 tokenizer=llama2 seeds=seed_206 dataset=all_domains_all_samples generation=temp_09 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained


