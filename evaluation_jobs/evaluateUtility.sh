#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --job-name=evaluateUtility
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --error=evaluateUtility.err
#SBATCH --output=evaluateUtility.out

python -m src.evaluation.calcUtility