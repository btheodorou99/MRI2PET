#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evaluateUtilityFull
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7
#SBATCH --error=evaluateUtilityFull_%A_%a.err
#SBATCH --output=evaluateUtilityFull_%A_%a.out

python -m src.evaluation.calcUtilityFullPiecewise --experiment_idx $SLURM_ARRAY_TASK_ID