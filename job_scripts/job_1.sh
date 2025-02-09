#!/bin/bash
#SBATCH --job-name=ml-1
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/logs/job_1.out
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/logs/job_1.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=47:45:00

echo "Starting job 1 at $(date)"
echo "Configuration: dataset=all_domains_10_samples prompt=number_list_answer_first_few_shot_structured train=number_list_answer_first_few_shot_structured ++train.training_args.per_device_train_batch_size=8 ++train.training_args.gradient_accumulation_steps=2"

# Training phase
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 training
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/train.py dataset=all_domains_10_samples prompt=number_list_answer_first_few_shot_structured train=number_list_answer_first_few_shot_structured ++train.training_args.per_device_train_batch_size=8 ++train.training_args.gradient_accumulation_steps=2
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 trained $TRAIN_EXIT
    
    # Get and run evaluation configs
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/run_evaluations.py 1
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 failed $TRAIN_EXIT
fi

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/check_queue.py

echo "Job 1 completed at $(date)"
