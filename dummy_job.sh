#!/bin/bash
#SBATCH --job-name=dummy
#SBATCH --output=dummy_%j.out
#SBATCH --error=dummy_%j.err
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

echo "Dummy job running"
sleep 10
echo "Dummy job completed"