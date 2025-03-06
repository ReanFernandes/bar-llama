#!/bin/bash
#SBATCH --job-name=ml-5
#SBATCH --output=/pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/logs/job_5.out
#SBATCH --error=/pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/logs/job_5.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=47:45:00
echo "Starting job 5 at $(date)"
echo "Configuration: model=llama2 tokenizer=llama2 dataset=all_domains_1_samples prompt=json_answer_first_zero_shot_structured train=json_answer_first_zero_shot_structured ++train.training_args.per_device_train_batch_size=7 ++train.training_args.gradient_accumulation_steps=1"

export HUGGINGFACE_TOKEN="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"
export WANDB_PROJECT=Final_runs_paper
# Training phase
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_status.py 5 training
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/train.py model=llama2 tokenizer=llama2 dataset=all_domains_1_samples prompt=json_answer_first_zero_shot_structured train=json_answer_first_zero_shot_structured ++train.training_args.per_device_train_batch_size=7 ++train.training_args.gradient_accumulation_steps=1
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_status.py 5 trained $TRAIN_EXIT
    
    # Run evaluations
    echo "Starting evaluations for job 5"
    
echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_1 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_206 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=greedy evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_025 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=trained"
fi

echo "Running evaluation with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
# Update eval status to running
python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "running"

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/eval.py seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained
EVAL_EXIT=$?

if [ $EVAL_EXIT -eq 0 ]; then
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "completed" $EVAL_EXIT
else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_eval_status.py 5 "seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained" "failed" $EVAL_EXIT "Evaluation failed"
    echo "Evaluation failed with config: seeds=seed_989 model=llama2 tokenizer=llama2 dataset=all_domains_1_samples generation=temp_06 evaluation_dataset=test_set_2 eval=json_answer_first_zero_shot_structured ++eval.quantisation_status=full_model ++eval.training_status=untrained"
fi

else
    python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/update_status.py 5 failed $TRAIN_EXIT
fi

python3 /pfs/work7/workspace/scratch/fr_rf1031-model_temp_storage/bar-llama/scripts/check_queue.py

echo "Job 5 completed at $(date)"
