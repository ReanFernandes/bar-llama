#!/bin/bash
#SBATCH --job-name=perm_test
#SBATCH --output=perm_test_%j.out
#SBATCH --error=perm_test_%j.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00       # Just 5 minutes
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

echo "Starting permission test at $(date)"
echo "Running on node: $(hostname)"

echo -e "\n1. Testing squeue access:"
squeue -u $USER
echo "squeue exit code: $?"

echo -e "\n2. Testing job submission:"
sbatch dummy_job.sh
echo "sbatch exit code: $?"

echo -e "\nTest completed at $(date)"

