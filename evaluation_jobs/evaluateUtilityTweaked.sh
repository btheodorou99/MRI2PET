#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evaluateUtilityTweaked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=evaluateUtilityTweaked.err
#SBATCH --output=evaluateUtilityTweaked.out

python -m src.evaluation.calcUtilityTweaked