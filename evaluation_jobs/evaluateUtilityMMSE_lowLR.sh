#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evaluateUtilityMMSE_lowLR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=evaluateUtilityMMSE_lowLR.err
#SBATCH --output=evaluateUtilityMMSE_lowLR.out

python -m src.evaluation.calcUtilityMMSE_lowLR