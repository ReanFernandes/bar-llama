#!/bin/bash
#SBATCH --job-name=ml-1
#SBATCH --output=/home/fr/fr_fr/fr_rf1031/bar-llama/logs/job_1.out
#SBATCH --error=/home/fr/fr_fr/fr_rf1031/bar-llama/logs/job_1.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=47:45:00
echo "Starting job 1 at $(date)"
echo "Configuration: dataset=all_domains_all_samples prompt=markdown_fact_first_few_shot_structured train=markdown_fact_first_few_shot_structured ++train.training_args.per_device_train_batch_size=8 ++train.training_args.gradient_accumulation_steps=2"

# Training phase
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 training
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/train.py dataset=all_domains_all_samples prompt=markdown_fact_first_few_shot_structured train=markdown_fact_first_few_shot_structured ++train.training_args.per_device_train_batch_size=8 ++train.training_args.gradient_accumulation_steps=2
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 trained $TRAIN_EXIT
    
    # Run evaluations
    echo "Starting evaluations for job 1"
    
echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_1 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=greedy evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_025 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/eval.py seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_eval_status.py 1 "seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 dataset=all_domains_all_samples generation=temp_06 evaluation_dataset=test_set_2 eval=markdown_fact_first_few_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

else
    python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/update_status.py 1 failed $TRAIN_EXIT
fi

python3 /home/fr/fr_fr/fr_rf1031/bar-llama/scripts/check_queue.py

echo "Job 1 completed at $(date)"
