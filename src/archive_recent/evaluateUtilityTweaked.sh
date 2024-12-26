#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=12
#SBATCH --job-name=evaluateUtilityTweaked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=evaluateUtilityFinal.err
#SBATCH --output=evaluateUtilityFinal.out

python -m src.evaluation.calcUtilityTweaked